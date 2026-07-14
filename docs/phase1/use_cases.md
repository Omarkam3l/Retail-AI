# Use Cases - Retail AI Surveillance Platform

This document describes the MVP and future use cases for the Retail AI Surveillance Platform.

---

## 1. MVP Use Cases (Phase 1)

### UC-1.1: Shoplifting & Item Pocketing Detection
* **Description**: Detect when a customer takes an item from a shelf and directly pockets it (jacket, pants, or shirt).
* **Actor**: System (automated analyzer), Store Associate.
* **Flow**:
  1. The customer picks up an item from the shelf.
  2. The system tracks the item bounding box and the customer's hands.
  3. The system detects the hand moving toward a pocket region (using pose estimation/keypoints).
  4. The item disappears inside the pocket boundary, and the hand emerges empty.
  5. The system triggers a high-priority "Concealment" alert.

### UC-1.2: Backpack & Purse Concealment
* **Description**: Detect when a customer places merchandise into a personal backpack, purse, or reusable shopping bag instead of a store-provided basket/cart.
* **Actor**: System, Store Associate.
* **Flow**:
  1. The customer positions their body or shopping cart to block direct lines of sight (using body orientation analysis).
  2. The customer transfers items from the shelf directly into a personal backpack or handbag.
  3. The system tracks the intersection of the product bounding box with the bag opening.
  4. The system triggers a "Bag Concealment" alert.

### UC-1.3: Loitering Detection in High-Value Zones
* **Description**: Identify individuals hovering in high-shrink aisles (e.g., alcohol, cosmetics, infant formula) for an extended period without picking up items or moving.
* **Actor**: System, Store Manager.
* **Flow**:
  1. A person enters a designated "High-Value Zone" (configured via polygons in the dashboard).
  2. The system tracks the duration of the person's presence.
  3. If the duration exceeds a threshold (e.g., 180 seconds) with minimal translation movement, the system flags the behavior.
  4. A low-priority "Loitering Alert" is dispatched to encourage a staff member to offer customer service (which naturally deters shoplifting).

### UC-1.4: Restricted Area Intrusion Detection
* **Description**: Detect unauthorized entry into employee-only zones (e.g., stockrooms, manager's office, safe rooms).
* **Actor**: System, Store Manager.
* **Flow**:
  1. A person crosses a virtual line (tripwire) or enters a restricted polygon.
  2. The system checks the time of day and employee indicators (e.g., high-visibility vests or badge detection, if configured).
  3. If unauthorized, the system instantly triggers an "Intrusion Alert".

### UC-1.5: Real-Time Alerts & Dispatch
* **Description**: Deliver suspicious activity alerts to store staff within 5 seconds of occurrence.
* **Actor**: System, Store Associate.
* **Flow**:
  1. The Behavior Analysis engine detects an alert event (UC-1.1 to UC-1.4).
  2. The system packages the event: Camera ID, Timestamp, Event Type, and a 3-second looping GIF or MP4 video clip showing the action.
  3. The system pushes the alert via WebSockets to the web dashboard and via Firebase Cloud Messaging (FCM) to associate mobile apps or smartwatches.
  4. The associate receives a vibration alert, views the video clip, and decides to intervene (e.g., "Hi, can I help you find a basket for those items?").

### UC-1.6: Evidence Clip Generation & Archiving
* **Description**: Maintain a secure, searchable archive of confirmed theft events for police reporting or insurance claims.
* **Actor**: Store Manager, System Administrator.
* **Flow**:
  1. Following an alert, the manager marks the event as "Confirmed Theft" in the dashboard.
  2. The system automatically saves a high-quality video clip (10 seconds before and 5 seconds after the event) to secure cloud storage.
  3. The system stamps the clip with: Date, Time, Store ID, Camera ID, and detected event metadata.
  4. The manager can export the file as a signed, tamper-evident MP4.

---

## 2. Future Use Cases (Phase 2 & 3 Roadmaps)

### UC-2.1: Violence & Altercation Detection
* **Description**: Detect physical fights, aggressive posturing, or weapon brandishing inside the store to protect staff and shoppers.
* **Model Requirement**: Spatio-temporal action transformers (e.g., VideoMAE, SlowFast) trained on aggressive physical actions.

### UC-2.2: Queue Monitoring & Checkout Optimization
* **Description**: Analyze checkout queue lengths and waiting times to alert managers when they need to open additional registers.
* **Model Requirement**: Object detection (people) + line crossing counts at cash registers.

### UC-2.3: In-Store Heatmaps & Customer Flow Analysis
* **Description**: Aggregate customer walking paths to identify high-traffic zones, dead corners, and optimal product placement areas.
* **Model Requirement**: Anonymized centroid tracking + coordinate mapping.

### UC-2.4: Real-time Shelf Out-of-Stock Monitoring
* **Description**: Detect empty shelf spaces for high-demand products and alert inventory staff to replenish shelves from the stockroom.
* **Model Requirement**: Fine-grained object detection of products on shelves.

### UC-2.5: Customer Demographic & Sentiment Analytics
* **Description**: Anonymously analyze broad customer groups (age range, gender presentation) and emotional responses to promotional displays.
* **Model Requirement**: Facial expression classification (non-biometric, processed on-edge, not stored).

### UC-2.6: Employee Compliance & Productivity Analytics
* **Description**: Verify if employees are wearing safety gear, performing scheduled cleaning tasks, or leaving registers unattended.
* **Model Requirement**: Pose estimation + task completion checklists.
