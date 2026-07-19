# Catalog Design

The product catalog holds SKU, Brand, Category, Reference images path, and the float32 embedding vector.

## Schema
```json
{
  "version": 1,
  "products": [
    {
      "sku": "cola-330ml",
      "name": "Coca-Cola 330ml",
      "brand": "Coca-Cola",
      "category": "beverages",
      "embedding": [0.012, -0.045, 0.89],
      "reference_images": ["data/catalog/cola.jpg"],
      "metadata": {}
    }
  ]
}
```
