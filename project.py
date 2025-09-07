import numpy as np
import streamlit as st
import cv2


# دالة تحميل الصورة الافتراضية أو المرفوعة

def get_uploaded_image_cv2(key=None):
    uploaded = st.file_uploader("📤 ارفع صورة (JPG/PNG)", type=["jpg", "jpeg", "png"], key=key)
    if uploaded is None:
        st.info("⬆️ رجاءً ارفع صورة للمتابعة.")
        st.stop()

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("❌ فشل في قراءة الصورة. تأكد من أن الملف صورة صالحة.")
        st.stop()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def lecture1_intro():
    st.title("📷 المحاضرة 1: مدخل ومعمارية الصور الرقمية")

    img_array = get_uploaded_image_cv2()  # رفع إلزامي
    st.markdown("""
          **ما هي الصورة الرقمية؟**
          - الصورة الرقمية عبارة عن مصفوفة من البكسلات (Pixels).
          - كل بكسل يمثل لونًا محددًا بناءً على قيم القنوات (Channels).
          - الأبعاد: `Height × Width × Channels`.
          - العمق اللوني (Bit Depth) يحدد عدد الألوان الممكنة.
          """)
    st.image(img_array, caption="الصورة الحالية", use_column_width=True)

    height, width = img_array.shape[:2]
    channels = 1 if img_array.ndim == 2 else img_array.shape[2]
    bit_depth = img_array.dtype.itemsize * 8

    st.write(f"**العرض:** {width} بكسل")
    st.write(f"**الطول:** {height} بكسل")
    st.write(f"**عدد القنوات:** {channels}")
    st.write(f"**العمق اللوني:** {bit_depth} بت لكل قناة")


def lecture2_colors():
    st.title("🎨 المحاضرة 2: أنظمة الألوان + فصل الألوان")
    img_array = get_uploaded_image_cv2()

    st.markdown("""
    **الشرح النظري:**
    - **RGB**: الأحمر، الأخضر، الأزرق.
    - **Grayscale**: تدرج الرمادي.
    - **HSV**: درجة اللون، التشبع، الإضاءة.
    - **Color Channel Separation**: فصل كل قناة لونية على حدة.
    """)

    # التحويلات
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # فصل القنوات
    R, G, B = cv2.split(img_array)

    # عرض الصور
    st.subheader("📌 أنظمة الألوان")
    col1, col2, col3 = st.columns(3)
    col1.image(img_array, caption="RGB", use_column_width=True)
    col2.image(gray, caption="Grayscale", use_column_width=True)
    col3.image(hsv, caption="HSV", use_column_width=True)

    st.subheader("📌 فصل الألوان (Color Channels)")
    col4, col5, col6 = st.columns(3)
    col4.image(R, caption="القناة الحمراء (R)", use_column_width=True)
    col5.image(G, caption="القناة الخضراء (G)", use_column_width=True)
    col6.image(B, caption="القناة الزرقاء (B)", use_column_width=True)

    # اختيار أي نسخة للحفظ
    save_option = st.selectbox(
        "اختر النسخة التي تريد حفظها:",
        ["RGB", "Grayscale", "HSV", "Red Channel", "Green Channel", "Blue Channel"]
    )

    if st.button("💾 حفظ النتيجة"):
        if save_option == "RGB":
            cv2.imwrite("lec2_rgb.png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        elif save_option == "Grayscale":
            cv2.imwrite("lec2_gray.png", gray)
        elif save_option == "HSV":
            cv2.imwrite("lec2_hsv.png", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        elif save_option == "Red Channel":
            cv2.imwrite("lec2_red.png", R)
        elif save_option == "Green Channel":
            cv2.imwrite("lec2_green.png", G)
        elif save_option == "Blue Channel":
            cv2.imwrite("lec2_blue.png", B)
        st.success(f"✅ تم حفظ نسخة {save_option} بنجاح")
def lecture3_point_ops():
    st.title("💡 المحاضرة 3: العمليات على البكسل")
    img_array = get_uploaded_image_cv2()

    brightness = st.slider("السطوع", -100, 100, 0)
    contrast = st.slider("التباين", -100, 100, 0)

    adjusted = cv2.convertScaleAbs(img_array, alpha=1 + contrast / 100, beta=brightness)

    if st.checkbox("تطبيق Negative"):
        adjusted = 255 - adjusted

    th_type = st.selectbox("Thresholding:", ["بدون", "بسيط", "Otsu"])
    if th_type != "بدون":
        gray = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)
        if th_type == "بسيط":
            _, adjusted = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            _, adjusted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    col1, col2 = st.columns(2)
    col1.image(img_array, caption="قبل", use_column_width=True)
    col2.image(adjusted, caption="بعد", use_column_width=True)

    if st.button("💾 حفظ النتيجة"):
        save_name = "lec3_result.png"
        if adjusted.ndim == 3:
            cv2.imwrite(save_name, cv2.cvtColor(adjusted, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(save_name, adjusted)
        st.success(f"✅ تم حفظ الصورة باسم {save_name}")


def lecture4_filters():
    st.title("🧪 المحاضرة 4: الفلاتر والالتفاف")
    img_array = get_uploaded_image_cv2()

    ftype = st.selectbox("نوع الفلتر:", ["Sharpen", "Gaussian Blur", "Median Blur", "Edge Detection"])
    k = st.slider("حجم Kernel (فردي)", 1, 15, 3, step=2)

    if ftype == "Sharpen":
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        filtered = cv2.filter2D(img_array, -1, kernel)
    elif ftype == "Gaussian Blur":
        filtered = cv2.GaussianBlur(img_array, (k, k), 0)
    elif ftype == "Median Blur":
        filtered = cv2.medianBlur(img_array, k)
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        filtered = cv2.Canny(gray, 100, 200)

    col1, col2 = st.columns(2)
    col1.image(img_array, caption="قبل", use_column_width=True)
    col2.image(filtered, caption=f"بعد {ftype}", use_column_width=True)

    if st.button("💾 حفظ النتيجة"):
        save_name = f"lec4_{ftype.replace(' ','_')}.png"
        if filtered.ndim == 3:
            cv2.imwrite(save_name, cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(save_name, filtered)
        st.success(f"✅ تم حفظ الصورة باسم {save_name}")

def lecture5_denoising():
    st.title("🔕 المحاضرة 5: إزالة الضوضاء")
    img_array = get_uploaded_image_cv2()

    noise_type = st.selectbox("نوع الضوضاء:", ["Salt & Pepper", "Gaussian"])
    noisy_img = img_array.copy()

    if noise_type == "Salt & Pepper":
        prob = st.slider("نسبة الضوضاء", 0.0, 0.1, 0.02)
        black = [0, 0, 0]
        white = [255, 255, 255]
        probs = np.random.rand(noisy_img.shape[0], noisy_img.shape[1])
        noisy_img[probs < (prob / 2)] = black
        noisy_img[probs > 1 - (prob / 2)] = white

    elif noise_type == "Gaussian":
        var = st.slider("قوة الضوضاء", 0.0, 50.0, 10.0)
        sigma = var ** 0.5
        gauss = np.random.normal(0, sigma, noisy_img.shape).astype('float32')
        noisy_img = np.clip(noisy_img.astype('float32') + gauss, 0, 255).astype('uint8')

    filter_type = st.selectbox("فلتر الإزالة:", ["Median", "Bilateral", "Gaussian"])
    k = st.slider("حجم Kernel", 1, 15, 3, step=2)

    if filter_type == "Median":
        processed = cv2.medianBlur(noisy_img, k)
    elif filter_type == "Bilateral":
        processed = cv2.bilateralFilter(noisy_img, d=k, sigmaColor=75, sigmaSpace=75)
    else:
        processed = cv2.GaussianBlur(noisy_img, (k, k), 0)

    col1, col2, col3 = st.columns(3)
    col1.image(img_array, caption="الأصلية", use_column_width=True)
    col2.image(noisy_img, caption="مع الضوضاء", use_column_width=True)
    col3.image(processed, caption="بعد الإزالة", use_column_width=True)

    if st.button("💾 حفظ النتيجة"):
        save_name = f"lec5_{noise_type}_{filter_type}.png"
        if processed.ndim == 3:
            cv2.imwrite(save_name, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(save_name, processed)
        st.success(f"✅ تم حفظ الصورة باسم {save_name}")

def lecture6_edges():
    st.title("✂️ المحاضرة 6: كشف الحواف + الكونتورز")
    img_array = get_uploaded_image_cv2()

    method = st.selectbox("طريقة الحواف:", ["Sobel", "Laplacian", "Canny"])
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    if method == "Sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))
    elif method == "Laplacian":
        edges = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
    else:
        t1 = st.slider("Threshold1", 0, 255, 100)
        t2 = st.slider("Threshold2", 0, 255, 200)
        edges = cv2.Canny(gray, t1, t2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_img = img_array.copy()
    cv2.drawContours(cont_img, contours, -1, (0,255,0), 2)

    col1, col2, col3 = st.columns(3)
    col1.image(img_array, caption="الأصلية", use_column_width=True)
    col2.image(edges, caption=f"حواف ({method})", use_column_width=True)
    col3.image(cont_img, caption="الكونتورز", use_column_width=True)

    if st.button("💾 حفظ النتيجة"):
        save_name = f"lec6_{method}_contours.png"
        cv2.imwrite(save_name, cv2.cvtColor(cont_img, cv2.COLOR_RGB2BGR))
        st.success(f"✅ تم حفظ الصورة باسم {save_name}")
def lecture7_morph():
    st.title("🧱 المحاضرة 7: العمليات المورفولوجية")
    img_array = get_uploaded_image_cv2()

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    morph_type = st.selectbox("العملية:", ["Erosion", "Dilation", "Opening", "Closing"])
    k = st.slider("حجم Kernel", 1, 15, 3, step=2)
    kernel = np.ones((k, k), np.uint8)

    if morph_type == "Erosion":
        result = cv2.erode(binary, kernel, iterations=1)
    elif morph_type == "Dilation":
        result = cv2.dilate(binary, kernel, iterations=1)
    elif morph_type == "Opening":
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    else:
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    col1, col2 = st.columns(2)
    col1.image(binary, caption="ثنائية", use_column_width=True)
    col2.image(result, caption=f"بعد {morph_type}", use_column_width=True)

    if st.button("💾 حفظ النتيجة"):
        save_name = f"lec7_{morph_type}.png"
        cv2.imwrite(save_name, result)
        st.success(f"✅ تم حفظ الصورة باسم {save_name}")

def lecture8_geo():
    st.title("🔄 المحاضرة 8: التحويلات الهندسية")
    img_array = get_uploaded_image_cv2()
    h, w = img_array.shape[:2]

    transform_type = st.selectbox("اختر نوع التحويل:", ["Translation", "Rotation", "Scaling", "Flipping", "Cropping"])

    if transform_type == "Translation":
        tx = st.slider("الإزاحة X", -w, w, 0)
        ty = st.slider("الإزاحة Y", -h, h, 0)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        transformed = cv2.warpAffine(img_array, M, (w, h))

    elif transform_type == "Rotation":
        angle = st.slider("زاوية الدوران", -180, 180, 0)
        scale = st.slider("نسبة التكبير/التصغير", 0.1, 3.0, 1.0)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
        transformed = cv2.warpAffine(img_array, M, (w, h))

    elif transform_type == "Scaling":
        scale = st.slider("النسبة", 0.1, 3.0, 1.0)
        transformed = cv2.resize(img_array, None, fx=scale, fy=scale)

    elif transform_type == "Flipping":
        flip_dir = st.radio("اتجاه الانعكاس:", ["أفقي", "رأسي", "كلاهما"])
        if flip_dir == "أفقي":
            transformed = cv2.flip(img_array, 1)
        elif flip_dir == "رأسي":
            transformed = cv2.flip(img_array, 0)
        else:
            transformed = cv2.flip(img_array, -1)

    elif transform_type == "Cropping":
        x1 = st.slider("X1", 0, w-1, 0)
        y1 = st.slider("Y1", 0, h-1, 0)
        x2 = st.slider("X2", x1+1, w, w)
        y2 = st.slider("Y2", y1+1, h, h)
        transformed = img_array[y1:y2, x1:x2]

    col1, col2 = st.columns(2)
    col1.image(img_array, caption="الأصلية", use_column_width=True)
    col2.image(transformed, caption=f"بعد {transform_type}", use_column_width=True)

    if st.button("💾 حفظ النتيجة"):
        save_name = f"lec8_{transform_type}.png"
        cv2.imwrite(save_name, cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))
        st.success(f"✅ تم حفظ الصورة باسم {save_name}")


def lecture9_feature():
    st.title("🏁 المحاضرة 9:استخراج الميزات")
    img_array = get_uploaded_image_cv2()
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    st.markdown("""
    **الشرح النظري:**
    - **ORB**: سريع وفعال، بديل مجاني لـ SIFT/SURF.
    - **FAST Corner**: يكتشف الزوايا بسرعة.
    - **HOG**: يصف شكل الجسم بناءً على اتجاهات الحواف.
    - **LoG**: يكشف الحواف باستخدام Gaussian + Laplacian.
    - **DoG**: يكشف الحواف باستخدام فرق Gaussian.
    - **SIFT**: يكتشف ويوصف النقاط المميزة بثبات أمام التغيير في الحجم والدوران.
    """)

    feature_method = st.selectbox("اختر الخوارزمية:", ["ORB", "FAST Corner", "HOG", "LoG", "DoG", "SIFT"])
    feature_img = img_array.copy()

    if feature_method == "ORB":
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(gray, None)
        feature_img = cv2.drawKeypoints(img_array, kp, None, color=(0,255,0))

    elif feature_method == "FAST Corner":
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(gray, None)
        feature_img = cv2.drawKeypoints(img_array, kp, None, color=(255,0,0))

    elif feature_method == "HOG":
        hog = cv2.HOGDescriptor()
        h = hog.compute(gray)
        st.write(f"HOG feature vector length: {len(h)}")
        feature_img = img_array  # HOG لا يرسم نقاط

    elif feature_method == "LoG":
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        log = cv2.Laplacian(blur, cv2.CV_64F)
        feature_img = cv2.convertScaleAbs(log)

    elif feature_method == "DoG":
        g1 = cv2.GaussianBlur(gray, (5,5), 0)
        g2 = cv2.GaussianBlur(gray, (9,9), 0)
        dog = cv2.absdiff(g1, g2)
        feature_img = dog

    elif feature_method == "SIFT":
        try:
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(gray, None)
            feature_img = cv2.drawKeypoints(img_array, kp, None, color=(0,255,255))
        except Exception:
            st.warning("⚠️ SIFT غير متاح في نسختك. ثبت opencv-contrib-python.")

    col1, col2 = st.columns(2)
    col1.image(img_array, caption="الأصلية", use_column_width=True)
    col2.image(feature_img, caption=f"ميزات ({feature_method})", use_column_width=True)

    if st.button("💾 حفظ النتيجة"):
        save_name = f"lec9_{feature_method.replace(' ', '_')}.png"
        if feature_img.ndim == 3:
            cv2.imwrite(save_name, cv2.cvtColor(feature_img, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(save_name, feature_img)
        st.success(f"✅ تم حفظ الصورة باسم {save_name}")

def lecture10_final():
    st.title("🏁 المحاضرة 10: المشروع الختامي (كل العمليات)")

    # رفع الصورة بنفس أسلوب باقي المحاضرات
    img_array = get_uploaded_image_cv2()
    processed = img_array.copy()

    st.subheader("🔄 التحويلات الهندسية")
    if st.checkbox("تطبيق Translation"):
        h, w = processed.shape[:2]
        tx = st.slider("الإزاحة X", -w, w, 0)
        ty = st.slider("الإزاحة Y", -h, h, 0)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        processed = cv2.warpAffine(processed, M, (w, h))

    if st.checkbox("تطبيق Rotation"):
        h, w = processed.shape[:2]
        angle = st.slider("زاوية الدوران", -180, 180, 0)
        scale = st.slider("نسبة التكبير/التصغير", 0.1, 3.0, 1.0)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
        processed = cv2.warpAffine(processed, M, (w, h))

    if st.checkbox("تطبيق Scaling"):
        scale = st.slider("النسبة", 0.1, 3.0, 1.0)
        processed = cv2.resize(processed, None, fx=scale, fy=scale)

    if st.checkbox("تطبيق Flipping"):
        flip_dir = st.radio("اتجاه الانعكاس:", ["أفقي", "رأسي", "كلاهما"])
        if flip_dir == "أفقي":
            processed = cv2.flip(processed, 1)
        elif flip_dir == "رأسي":
            processed = cv2.flip(processed, 0)
        else:
            processed = cv2.flip(processed, -1)

    if st.checkbox("تطبيق Cropping"):
        h, w = processed.shape[:2]
        x1 = st.slider("X1", 0, w-1, 0)
        y1 = st.slider("Y1", 0, h-1, 0)
        x2 = st.slider("X2", x1+1, w, w)
        y2 = st.slider("Y2", y1+1, h, h)
        processed = processed[y1:y2, x1:x2]

    st.subheader("🛠 المعالجات")
    if st.checkbox("تحويل إلى Gray"):
        processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)

    if st.checkbox("Gaussian Blur"):
        processed = cv2.GaussianBlur(processed, (5, 5), 0)

    if st.checkbox("Median Blur"):
        processed = cv2.medianBlur(processed, 3)

    if st.checkbox("Canny Edge Detection"):
        if processed.ndim == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        processed = cv2.Canny(processed, 100, 200)

    if st.checkbox("Sobel Edge Detection"):
        if processed.ndim == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(processed, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(processed, cv2.CV_64F, 0, 1, ksize=3)
        processed = cv2.magnitude(sobelx, sobely)
        processed = cv2.convertScaleAbs(processed)

    if st.checkbox("Laplacian Edge Detection"):
        if processed.ndim == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        processed = cv2.Laplacian(processed, cv2.CV_64F)
        processed = cv2.convertScaleAbs(processed)

    if st.checkbox("Thresholding"):
        if processed.ndim == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        _, processed = cv2.threshold(processed, 127, 255, cv2.THRESH_BINARY)

    st.subheader("📷 النتيجة")
    col1, col2 = st.columns(2)
    col1.image(img_array, caption="الصورة الأصلية", use_column_width=True)
    col2.image(processed, caption="بعد المعالجة", use_column_width=True)

    if st.button("💾 حفظ النتيجة النهائية"):
        if processed.ndim == 3:
            cv2.imwrite("final_project_result.png", cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite("final_project_result.png", processed)
        st.success("✅ تم حفظ النتيجة باسم final_project_result.png")
def main_menu():
    st.sidebar.title("📚 قائمة المحاضرات")
    choice = st.sidebar.radio("اختر المحاضرة:", [
        "1️⃣ مدخل ومعمارية الصور الرقمية",
        "2️⃣ أنظمة الألوان",
        "3️⃣ العمليات على البكسل",
        "4️⃣ الفلاتر والالتفاف",
        "5️⃣ إزالة الضوضاء",
        "6️⃣ كشف الحواف",
        "7️⃣ العمليات المورفولوجية",
        "8️⃣ التحويلات الهندسية",
        "9️⃣  استخراج الميزات",
        "تطبيق كافة المفاهيم"
    ])
    if choice.startswith("1"):
        lecture1_intro()
    elif choice.startswith("2"):

        lecture2_colors()
    elif choice.startswith("3"):
        lecture3_point_ops()
    elif choice.startswith("4"):
        lecture4_filters()
    elif choice.startswith("5"):
        lecture5_denoising()
    elif choice.startswith("6"):
        lecture6_edges()
    elif choice.startswith("7"):
        lecture7_morph()
    elif choice.startswith("8"):
        lecture8_geo()
    elif choice.startswith("9"):
        lecture9_feature()
    elif choice == "تطبيق كافة المفاهيم":
        lecture10_final()




def welcome_page():
    st.title("📸 سلسلة محاضرات تفاعلية في معالجة الصور")
    st.markdown("""
    أهلاً بك في مشروع **معالجة الصور الرقمية** 🎯

    هذا التطبيق التعليمي يشرح أهم عمليات معالجة الصور خطوة بخطوة.
    - كل محاضرة تحتوي على شرح نظري مختصر.
    - تجربة عملية تفاعلية بدون كتابة أي كود.
    - يمكنك رفع صورة أو استخدام صورة جاهزة.

    **ابدأ رحلتك الآن وتعرف على عالم معالجة الصور!**
    """)
    if st.button("🚀 ابدأ المحاضرات"):
        st.session_state.page = "lectures"


if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if st.session_state.page == "welcome":
        welcome_page()
    elif st.session_state.page == "lectures":
        main_menu()


