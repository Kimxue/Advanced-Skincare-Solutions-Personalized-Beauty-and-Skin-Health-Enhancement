from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Đường dẫn đến mô hình đã huấn luyện
model_path = 'best_model.tf'
model = tf.saved_model.load(model_path)

# Đọc file CSV chứa thông tin sản phẩm
df_products = pd.read_csv('skincare_products_clean.csv')

# Xác định các thành phần phù hợp cho từng loại da
ingredients_by_skin_type = {
    'dry': ['glycerin', 'hyaluronic acid', 'ceramides'],
    'oily': ['salicylic acid', 'niacinamide'],
    'normal': ['glycerin', 'hyaluronic acid', 'niacinamide']
}

# Hàm dự đoán loại da từ ảnh
def predict_skin_type(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model(img_array)
    class_names = ['dry', 'normal', 'oily']
    skin_type = class_names[np.argmax(prediction)]

    return skin_type

# Gợi ý sản phẩm
def recommend_products(skin_type):
    product_types = df_products['product_type'].unique()
    recommended_products = []

    for p_type in product_types:
        products = df_products[df_products['product_type'] == p_type]
        for index, row in products.iterrows():
            ingredients = eval(row['clean_ingreds'])
            if any(ingredient in ingredients for ingredient in ingredients_by_skin_type[skin_type]):
                recommended_products.append({
                    'product_type': p_type,
                    'product_name': row['product_name'],
                    'product_url': row['product_url'],
                    'price': row['price']
                })
                break

    return recommended_products

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            img_path = os.path.join('static', file.filename)
            file.save(img_path)

            skin_type = predict_skin_type(img_path, model)
            recommendations = recommend_products(skin_type)

            return render_template('index.html', skin_type=skin_type.capitalize(), recommendations=recommendations, img_path=img_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
