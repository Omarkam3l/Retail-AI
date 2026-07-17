# API Design Specification - Phase 2 Design Review

This document specifies the REST API endpoints and real-time WebSocket protocol definitions for the Retail AI Surveillance Platform.

---

## 1. REST API Specification (FastAPI OpenAPI)

All requests must contain an `Authorization: Bearer <JWT_TOKEN>` header (except public auth endpoints).

### 1.1 Authentication Endpoints

#### POST `/api/v1/auth/login`
Authenticates a store associate or manager and returns a JWT token.
* **Request Body**:
  ```json
  {
    "email": "manager@store.com",
    "password": "securepassword123"
  }
  ```
* **Success Response (200 OK)**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsIn...",
    "token_type": "bearer",
    "user": {
      "id": "a6b1076b-9c71-47cc-98fc-1200fa44766f",
      "email": "manager@store.com",
      "role": "Manager",
      "tenant_id": "b10a2731-0cfd-4d7a-be00-ff995874220a"
    }
  }
  ```

---

### 1.2 Edge-to-Cloud Alert Ingestion

#### POST `/api/v1/alerts`
Triggered by the Edge Node to report a behavior event. Sent as a multi-part form request containing JSON metadata and a binary video file.
* **Headers**: `X-Edge-Auth-Token: <UUID_TOKEN>` (Edge node specific validation token).
* **Multipart Fields**:
  - `metadata`: JSON string
    ```json
    {
      "camera_id": "78bcf10c-99a3-4876-90fe-0012fabcde34",
      "event_type": "Pocket Concealment",
      "timestamp": "2026-07-17T16:00:26Z"
    }
    ```
  - `file`: Binary H.264 MP4 file stream (<400KB).
* **Success Response (201 Created)**:
  ```json
  {
    "alert_id": "99b0cde2-50d4-4a2a-b0fe-bcde12345678",
    "clip_url": "https://s3.amazonaws.com/retail-ai-clips/99b0cde2-50d4-4a2a-b0fe-bcde12345678.mp4",
    "status": "Queued"
  }
  ```

---

### 1.3 Associate Feedback Loop

#### POST `/api/v1/alerts/{alert_id}/feedback`
Submitted by store associates to log resolution action and validate detection accuracy.
* **Path Parameters**: `alert_id` (UUID)
* **Request Body**:
  ```json
  {
    "status": "True Positive", -- Check in ('True Positive', 'False Positive', 'Prevented')
    "notes": "Customer returned items after associate approached."
  }
  ```
* **Success Response (200 OK)**:
  ```json
  {
    "status": "Logged",
    "alert_id": "99b0cde2-50d4-4a2a-b0fe-bcde12345678"
  }
  ```

---

### 1.4 Camera Stream Configuration

#### POST `/api/v1/cameras/{camera_id}/config`
Updates stream boundaries and detection thresholds (e.g. loitering timers, restricted area polygon zones).
* **Request Body**:
  ```json
  {
    "loiter_threshold_seconds": 120,
    "intrusion_polygons": [
      {
        "zone_name": "Backroom Entrance",
        "coordinates": [[100, 200], [300, 200], [300, 500], [100, 500]]
      }
    ]
  }
  ```
* **Success Response (200 OK)**:
  ```json
  {
    "status": "ConfigUpdated",
    "camera_id": "78bcf10c-99a3-4876-90fe-0012fabcde34"
  }
  ```

---

## 2. Real-Time Communication Protocol (WebSockets)

For push-alert distribution and configuration sync, we establish WebSocket servers.

```text
Dashboard / Mobile Client                 Cloud WebSocket Service
          │                                         │
          ├───────── Connection Request ───────────>│ (Verify JWT)
          │<──────── Connection Accepted ───────────┤
          │                                         │
          │<─── Alert Event Broadcast (Push JSON) ──┤ (Concealment detected)
          │                                         │
          ├───────── Client Ping ──────────────────>│
          │<──────── Server Pong ───────────────────┤
```

### 2.1 Alert Event Broadcast Payload (JSON)
Dispatched immediately when the cloud ingests an edge event. Sent to all active sockets belonging to the store's `tenant_id`.
```json
{
  "event": "new_alert",
  "data": {
    "alert_id": "99b0cde2-50d4-4a2a-b0fe-bcde12345678",
    "camera_name": "Aisle 3 - Liquor Shelf",
    "event_type": "Pocket Concealment",
    "timestamp": "2026-07-17T16:00:26Z",
    "clip_url": "https://s3.amazonaws.com/retail-ai-clips/99b0cde2-50d4-4a2a-b0fe-bcde12345678.mp4"
  }
}
```

### 2.2 Protocol Trade-offs: WebSockets vs Server-Sent Events (SSE)
* **WebSockets**:
  - *Pros*: Bi-directional (client can send keepalives, check statuses, or acknowledge alerts back over the same connection).
  - *Cons*: Higher state overhead on server; firewall traversal issues on legacy corporate networks.
* **Server-Sent Events (SSE)**:
  - *Pros*: Built-in automatic reconnection, text-only stream is extremely lightweight, uses standard HTTP.
  - *Cons*: Uni-directional only (client must issue separate HTTP requests to send feedback, which increases request overhead).
* **Decision**: **WebSockets**. The React dashboard and mobile applications require frequent two-way messaging (e.g. sync configurations, query camera streams, acknowledge alarms). Bi-directional connections reduce overall transaction overhead.
