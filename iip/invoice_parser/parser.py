import json
from invoice_parser.detect import detect_image
from invoice_parser.ocr import ocr_exrtact






def parse_invoice(img):
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

    detections=detect_image(img)
    print(detections)
    
    final_detections={'items':[{} for i in range(5)]}
    # print(final_detections)
    temp={}
    for i in detections['items']:
        temp_img=img.crop((i['x'],i['y'],i['x']+i['width'],i['y']+i['height']))
        print(i['label'])
        temp[i['label']]=ocr_exrtact(temp_img)
        # temp_img.show()
    
    for i in range(5):
        for j in temp.keys():
            st_len=int(len(temp[j])/5)
            final_detections['items'][i][j]=temp[j][i*st_len:(i+1)*st_len]
    return json.loads(json.dumps(final_detections))