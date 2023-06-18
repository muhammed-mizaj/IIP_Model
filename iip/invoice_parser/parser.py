import json
from invoice_parser.detect import run
from invoice_parser.ocr import ocr_exrtact
import os
import shutil
from img2table.ocr import TesseractOCR
# from img2table.document import Image
import pytesseract
from PIL import Image
from invoice_parser.extract_rows import get_rows

ocr = TesseractOCR(n_threads=1, lang="eng")

# Instantiation of document, either an image or a PDF


# Table extraction

def parse_invoice(img):
    example={
    "items": [
        {
        "itemname": "Bacon Turkey Breakfast strips came Meals Frozen 2kg UAE Product Code:FI120138",
        "unit": "KILOGRAM",
        "qty": 16.00,
        "price": 200.00,
        "itemno": "1"
        },
        {
            "itemname": "Turkey fresh bacon sliced",
            "unit": "KILOGRAM",
            "qty": 28.00,
            "price": 70.00,
            "itemno": "2"
        },
        {
            "itemname": "Frozen chicken prawn dumpling with coriander 20GMS",
            "unit": "KILOGRAM",
            "qty": 23.00,
            "price": 50.00,
            "itemno": "3"
        },
        {
            "itemname": "carne veal salumetti Al tartufo whole CHICKEN SIEW MAI 20GM SHITAKE",
            "unit": "KILOGRAM",
            "qty": 34.00,
            "price": 55.00,
            "itemno": "4"
        },
        {
            "itemname": "MUSHROOM WATERCHESTNUT carne wagyu beef bresaola whole kg",
            "unit": "KILOGRAM",
            "qty": 15.00,
            "price": 32.00,
            "itemno": "5"
        },
        {
            "itemname": "beef strips cooked smoked premium",
            "unit": "KILOGRAM",
            "qty": 21.00,
            "price": 43.00,
            "itemno": "6"
        },
        {
            "itemname": "american style frozen 2kg",
            "unit": "KILOGRAM",
            "qty": 22.00,
            "price": 30.00,
            "itemno": "7"
        },
             ]
    }

    run(weights='invoice_parser/weights/best.pt',source=img,save_crop=True,exist_ok=True)
    print("Croppped and Saved")
    main_dir='invoice_parser/runs/detect/exp/crops/'
    detections={}
    for i in os.listdir(main_dir):
        files=os.listdir(main_dir+i)
        print(main_dir+i+'/'+files[0])
        data=pytesseract.image_to_string(Image.open(main_dir+i+'/'+files[0]))
        data=data.split("\n")
        data=[i for i in data if i != ""]
        detections[i]=data

    print(detections)

    if(len(detections['items'])>len(detections['unit'])):
        for i in range(len(detections['items'])-len(detections['unit'])):
            if i%2==0:
                del detections['items'][0]
            else:
                del detections['items'][-1]

    # print(detections)

    new_detections={"items":[]}
    num_items = len(detections['items'])
    print(detections['items'])

    # Iterate over the items and create the new dictionary structure
    print(num_items)
    for i in range(num_items):
        try:
            item = {
            'itemname': detections['items'][i] if i < len(detections['items']) else "",
            'qty': detections['qty'][i] if i < len(detections['qty']) else "",
            'unit': detections['unit'][i] if i < len(detections['unit']) else "",
            'price':detections['price'][i] if i < len(detections['price']) else ""
            }
        except Exception as e:
            print(e)
        new_detections['items'].append(item)

    print(new_detections)



        

        

    shutil.rmtree('invoice_parser/runs/detect/exp/')

    
    print("Finished Running")
    return json.loads(json.dumps(new_detections))