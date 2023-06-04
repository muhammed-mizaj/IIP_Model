import json




def parse_invoice():
    example={
    "items": [
        {
        "itemname": "Item 1",
        "unit": "Piece",
        "qty": 10,
        "price": 5.99,
        "itemno": "A001"
        },
        {
        "itemname": "Item 2",
        "unit": "Pack",
        "qty": 5,
        "price": 12.99,
        "itemno": "B002"
        },
        {
        "itemname": "Item 3",
        "unit": "Box",
        "qty": 2,
        "price": 19.99,
        "itemno": "C003"
        }
        ]
    }
    return json.loads(json.dumps(example))