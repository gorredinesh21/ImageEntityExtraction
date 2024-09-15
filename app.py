import pandas as pd
from PIL import Image
import os
import pytesseract
import re
df=pd.read_csv('dataset/test.csv')
from pathlib import Path
df['image_name']=df['image_link'].apply(lambda link:Path(link).name)
# Given unit_suffix_map
unit_suffix_map = {
    'centimetre': {'cm', 'centimeter', 'centimetre'},
    'metre': {'m', 'meter', 'metre'},
    'millimetre': {'mm', 'millimeter', 'millimetre'},
    'foot': {'ft', 'foot'},
    'inch': {'in', 'inch'},
    'yard': {'yd', 'yard'},
    'gram': {'g', 'gram'},
    'kilogram': {'kg', 'kilogram'},
    'microgram': {'μg', 'microgram', 'mcg'},
    'milligram': {'mg', 'milligram'},
    'ounce': {'oz', 'ounce'},
    'pound': {'lb', 'pound'},
    'ton': {'t', 'ton'},
    'kilovolt': {'kV', 'kilovolt'},
    'millivolt': {'mV', 'millivolt'},
    'volt': {'V', 'volt'},
    'kilowatt': {'kW', 'kilowatt'},
    'watt': {'W', 'watt'},
    'centilitre': {'cl', 'centilitre', 'centiliter'},
    'cubic foot': {'cu ft', 'cubic foot'},
    'cubic inch': {'cu in', 'cubic inch'},
    'cup': {'cup'},
    'decilitre': {'dl', 'decilitre', 'deciliter'},
    'fluid ounce': {'fl oz', 'fluid ounce'},
    'gallon': {'gal', 'gallon'},
    'imperial gallon': {'imp gal', 'imperial gallon'},
    'litre': {'L', 'litre', 'liter'},
    'microlitre': {'μL', 'microlitre', 'microliter'},
    'millilitre': {'mL', 'millilitre', 'milliliter'},
    'pint': {'pt', 'pint'},
    'quart': {'qt', 'quart'}
}

def extract_values_and_units(text, unit_suffix_map):
    results = []

    # Create a regex pattern to find numbers followed by potential unit suffixes
    for unit, suffixes in unit_suffix_map.items():
        for suffix in suffixes:
            # Regex pattern to match a number followed by the unit (with optional space)
            pattern = r'(\d+\.?\d*)\s*(' + re.escape(suffix) + r')'
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            # Append matched number and unit
            for match in matches:
                number = match[0]  # The number
                found_unit = unit  # The full unit name
                results.append((number, found_unit))

    return results

entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'maximum_weight_recommendation': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'}
}

# Traverse through the directory

def final(entity_name1,image_name1):
    directory_path='test_images/'
    allowed_units=list(entity_unit_map[entity_name1])
    file_path = os.path.join(directory_path, image_name1)
    extracted_text=pytesseract.image_to_string(Image.open(file_path))
    extracted_values_units = extract_values_and_units(extracted_text, unit_suffix_map)
    if extracted_values_units:
        for number, unit in extracted_values_units:
                for i in allowed_units:
                     if unit==i:
                          xgf=round(float(number),1)
                          s=str(xgf)+' '+unit
                          return s 
                          
    else:
        return "No units and values found."
        
# Replace 'train_images' with your actual folder path
folder_path = 'test_images'


# Get a list of image filenames in the folder
existing_images = os.listdir(folder_path)

# Filter the DataFrame based on image existence
df2 = df[df['image_name'].isin(existing_images)]
df2['predicted_entity_value'] = None

# Apply the final function to each row with existing images
for index, row in df2.iterrows():
    image_name = row['image_name']
    entity_name = row['entity_name']

    if image_name in existing_images:  # Check if image exists
        try:
            predicted_value = final(entity_name, image_name)
            df2.at[index, 'predicted_entity_value'] = predicted_value
        except Exception as e:  # Handle potential errors during image processing
            print(f"Error processing image {image_name}: {e}")


df2.to_csv('generated_test.csv',index=False)

df2 = df2.rename(columns={'predicted_entity_value': 'prediction'})

# Replace NaN values with an empty string
df2['prediction'] = df2['prediction'].fillna('')

# Replace 'No units and values found.' with an empty string
df2['prediction'] = df2['prediction'].replace('No units and values found.', '')

# Ensure the column is of string type
df2['prediction'] = df2['prediction'].astype(str)

X=df2[['index', 'prediction']]


X.to_csv('output.csv',index=False)