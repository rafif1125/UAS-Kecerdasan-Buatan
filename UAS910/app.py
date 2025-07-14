import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.neighbors import KNeighborsClassifier

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# DATA SOAL 6
data = [
    [0.3, 7.21, 'Douglas Fir'],
    [0.18, 5.12, 'Douglas Fir'],
    [0.46, 8.83, 'Douglas Fir'],
    [0.63, 12.08, 'Douglas Fir'],
    [0.23, 5.81, 'Douglas Fir'],
    [0.56, 13.5, 'Douglas Fir'],
    [0.39, 10.9, 'Douglas Fir'],
    [0.41, 6.79, 'Douglas Fir'],
    [0.62, 10.66, 'Douglas Fir'],
    [0.43, 10.5, 'Douglas Fir'],
    [0.15, 2.67, 'Douglas Fir'],
    [0.19, 20.34, 'White Pine'],
    [0.17, 19.72, 'White Pine'],
    [0.17, 19.8, 'White Pine'],
    [0.22, 23.7, 'White Pine'],
    [0.45, 32.51, 'White Pine'],
    [0.39, 26.23, 'White Pine'],
    [0.42, 32.51, 'White Pine'],
    [0.38, 29.18, 'White Pine'],
    [0.3, 26.1, 'White Pine'],
    [0.18, 21.51, 'White Pine'],
]

# SISTEM PAKAR + CERTAINTY FACTOR
def expert_system_predict(diameter, tinggi):
    if diameter > 0.25 and tinggi > 20:
        return "White Pine", 0.85
    elif diameter <= 0.25 and tinggi <= 20:
        return "Douglas Fir", 0.75
    elif diameter > 0.25 and tinggi <= 20:
        return "Douglas Fir", 0.60
    elif diameter <= 0.25 and tinggi > 20:
        return "White Pine", 0.60
    else:
        return "Unknown", 0.5

# KNN TRAINING
X = [[d[0], d[1]] for d in data]
y = [d[2] for d in data]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # Max 2 MB

# Utility cek ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    method = 'manual'
    image_url = None

    if request.method == 'POST':
        method = request.form.get('method', 'manual')

        # ==== VISION MODE ====
        if method == 'vision':
            file = request.files.get('image')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Pastikan folder upload ada
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_url = url_for('static', filename='uploads/' + filename)
            else:
                image_url = None
            try:
                diameter = float(request.form['vision_diameter'])
                tinggi = float(request.form['vision_tinggi'])
            except Exception:
                result = {'error': 'Diameter/Tinggi (hasil vision) tidak valid.'}
                return render_template('index.html', result=result, method=method, image_url=image_url)
        else:
            try:
                diameter = float(request.form['diameter'])
                tinggi = float(request.form['tinggi'])
            except Exception as e:
                result = {'error': f'Input tidak valid: {e}'}
                return render_template('index.html', result=result, method=method, image_url=image_url)

        # ==== PREDIKSI ====
        pakar_pred, cf = expert_system_predict(diameter, tinggi)
        knn_pred = knn.predict([[diameter, tinggi]])[0]

        if pakar_pred == knn_pred:
            nilai_kebenaran = cf
            cocok = True
        else:
            nilai_kebenaran = 0
            cocok = False

        result = {
            'diameter': diameter,
            'tinggi': tinggi,
            'pakar_pred': pakar_pred,
            'cf': cf,
            'knn_pred': knn_pred,
            'cocok': cocok,
            'nilai_kebenaran': nilai_kebenaran
        }
    return render_template('index.html', result=result, method=method, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
