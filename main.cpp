#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QPainter>
#include <QImage>
#include <QSlider>
#include <QLabel>
#include <QComboBox>
#include <QCheckBox>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QTimer>
#include <QMessageBox>
#include <QFile>
#include <QTextStream>
#include <QKeyEvent>
#include <QDir>
#include <QCoreApplication>
#include <QStringList>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

// =========================
// VECTORES, COLORES Y MATRICES
// =========================

// Generamos un vector de 3 componentes para las posiciones, direcciones y normales
struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    // Se suman dos vectores.
    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }

    // Se restan dos vectores.
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }

    // Se multiplica el vector por un escalar.
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }

    // Dividimos el vector entre un escalar.
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }
};

// Creamos un vector de 4 componentes para el pipeline matricial
struct Vec4 {
    float x, y, z, w;
    Vec4() : x(0), y(0), z(0), w(0) {}
    Vec4(float X, float Y, float Z, float W) : x(X), y(Y), z(Z), w(W) {}
};

// Generamos los colores RGB en punto flotante
struct Color3 {
    float r, g, b;
    Color3() : r(0), g(0), b(0) {}
    Color3(float R, float G, float B) : r(R), g(G), b(B) {}

    // Se suman dos colores
    Color3 operator+(const Color3& o) const { return Color3(r + o.r, g + o.g, b + o.b); }

    // se multiplica un color por un escalar
    Color3 operator*(float s) const { return Color3(r * s, g * s, b * s); }
};

// Generamos la matriz 4x4 para lo que seria el modelo, vista, proyección y viewport
struct Mat4 {
    float m[4][4];
    Mat4() {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                m[r][c] = 0.0f;
    }
};

// =========================
// UTILIDADES MATEMÁTICAS BÁSICAS
// =========================

// Producto punto
static float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Producto cruz
static Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Longitud de un vector
static float lengthVec(const Vec3& v) {
    return std::sqrt(dot(v, v));
}

// Devolvemos el vector normalizado
static Vec3 normalize(const Vec3& v) {
    float len = lengthVec(v);
    if (len < 1e-8f) return Vec3(0, 0, 0);
    return v * (1.0f / len);
}

// Limitamos los rangos a 0 y 1
static float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

// Limitamos tambien el color a un rango visible
static Color3 clampColor(const Color3& c) {
    return Color3(clamp01(c.r), clamp01(c.g), clamp01(c.b));
}

// Se devuelve el mayor de tres valores
static float max3(float a, float b, float c) {
    return std::max(a, std::max(b, c));
}

// =========================
// MATRICES 4x4 DEL PIPELINE
// =========================

// Generamos la matriz identidad
static Mat4 identity() {
    Mat4 r;
    for (int i = 0; i < 4; ++i) r.m[i][i] = 1.0f;
    return r;
}

// Multplicamos dos matrices 4x4
static Mat4 multiplyMat4(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) sum += a.m[i][k] * b.m[k][j];
            r.m[i][j] = sum;
        }
    }
    return r;
}

// Multiplicamos una matriz 4x4 por un vector 4D
static Vec4 multiplyVec4(const Mat4& m, const Vec4& v) {
    return Vec4(
        m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3] * v.w,
        m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3] * v.w,
        m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3] * v.w,
        m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3] * v.w
    );
}

// Matriz de traslacion
static Mat4 translation(float tx, float ty, float tz) {
    Mat4 r = identity();
    r.m[0][3] = tx;
    r.m[1][3] = ty;
    r.m[2][3] = tz;
    return r;
}

// Matriz de escalamiento
static Mat4 scaling(float sx, float sy, float sz) {
    Mat4 r = identity();
    r.m[0][0] = sx;
    r.m[1][1] = sy;
    r.m[2][2] = sz;
    return r;
}

// Matriz de rotacion en x
static Mat4 rotationX(float a) {
    Mat4 r = identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[1][1] = c;  r.m[1][2] = -s;
    r.m[2][1] = s;  r.m[2][2] = c;
    return r;
}

// Matriz de rotacion en y
static Mat4 rotationY(float a) {
    Mat4 r = identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[0][0] = c;  r.m[0][2] = s;
    r.m[2][0] = -s; r.m[2][2] = c;
    return r;
}

// Matriz de rotacion en z
static Mat4 rotationZ(float a) {
    Mat4 r = identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[0][0] = c;  r.m[0][1] = -s;
    r.m[1][0] = s;  r.m[1][1] = c;
    return r;
}

// Hacemos una camara con un estilo lookAt
static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);

    Mat4 r = identity();
    r.m[0][0] = s.x;  r.m[0][1] = s.y;  r.m[0][2] = s.z;  r.m[0][3] = -dot(s, eye);
    r.m[1][0] = u.x;  r.m[1][1] = u.y;  r.m[1][2] = u.z;  r.m[1][3] = -dot(u, eye);
    r.m[2][0] = -f.x; r.m[2][1] = -f.y; r.m[2][2] = -f.z; r.m[2][3] = dot(f, eye);
    return r;
}

// Generamoms la matriz de proyeccion en perspectiva
static Mat4 perspective(float fovyRadians, float aspect, float zNear, float zFar) {
    Mat4 r;
    float t = std::tan(fovyRadians * 0.5f);
    float f = 1.0f / std::max(t, 1e-6f);
    r.m[0][0] = f / aspect;
    r.m[1][1] = f;
    r.m[2][2] = (zFar + zNear) / (zNear - zFar);
    r.m[2][3] = (2.0f * zFar * zNear) / (zNear - zFar);
    r.m[3][2] = -1.0f;
    return r;
}

// De igual forma lo hacemos con una matriz para la proyeccion ortogonal
static Mat4 orthographic(float left, float right, float bottom, float top, float zNear, float zFar) {
    Mat4 r = identity();
    r.m[0][0] = 2.0f / (right - left);
    r.m[1][1] = 2.0f / (top - bottom);
    r.m[2][2] = -2.0f / (zFar - zNear);
    r.m[0][3] = -(right + left) / (right - left);
    r.m[1][3] = -(top + bottom) / (top - bottom);
    r.m[2][3] = -(zFar + zNear) / (zFar - zNear);
    return r;
}

// Hacemos la matriz de viewport para pasar de NDC a pixeles
static Mat4 viewport(float width, float height) {
    Mat4 r = identity();
    r.m[0][0] = width * 0.5f;
    r.m[0][3] = width * 0.5f;
    r.m[1][1] = -height * 0.5f;
    r.m[1][3] = height * 0.5f;
    r.m[2][2] = 0.5f;
    r.m[2][3] = 0.5f;
    return r;
}

// Extraemos lo que son x, y, z de un vector 4D
static Vec3 xyz(const Vec4& v) {
    return Vec3(v.x, v.y, v.z);
}

// =========================
// ESTRUCTURAS DEL MODELO Y LA ESCENA
// =========================

// Vemos una esquina de la cara del OBJ
struct FaceVertex {
    int v = -1;
    int vn = -1;
};

// Por cada cara que es triangular guardamos tres esquinas
struct Face {
    FaceVertex a, b, c;
};

// Decidimos los materiales para el modelo phong
struct Material {
    Color3 ka;
    Color3 kd;
    Color3 ks;
    float shininess;
};

// Damos los datos de la camara
struct CameraData {
    Vec3 eye;
    QString name;
};

// Damos los datos de la luz
struct LightData {
    bool enabled;
    Vec3 pos;
    Color3 intensity;
};

// =========================
//      WIDGET DE RENDER
// =========================

class RenderWidget : public QWidget {
public:
    explicit RenderWidget(QWidget* parent = nullptr) : QWidget(parent) {
        // Tamaño mínimo y foco para captar teclas.
        setMinimumSize(920, 680);
        setFocusPolicy(Qt::StrongFocus);

        // LAs camara ya con un modelo normalizado
        cameras.push_back({Vec3(0.0f, 0.15f, 4.20f), "Camara 1"});
        cameras.push_back({Vec3(2.60f, 1.20f, 4.60f), "Camara 2"});
        cameras.push_back({Vec3(-3.00f, 1.70f, 3.80f), "Camara 3"});

        // Material A
        materials.push_back({
            Color3(0.00f, 0.00f, 0.00f),
            Color3(0.50f, 0.50f, 0.50f),
            Color3(0.70f, 0.70f, 0.70f),
            32.0f
        });

        // Material B
        materials.push_back({
            Color3(0.23125f, 0.23125f, 0.23125f),
            Color3(0.2775f, 0.2775f, 0.2775f),
            Color3(0.773911f, 0.773911f, 0.773911f),
            89.60f
        });

        // Luz blanca
        whiteLight = {true, Vec3(2.8f, 3.2f, 4.6f), Color3(1.0f, 1.0f, 1.0f)};

        // Luz azul
        blueLight  = {true, Vec3(-3.0f, 1.8f, 2.2f), Color3(0.20f, 0.35f, 0.90f)};
    }

    // CArgamos el OBJ, leemos los vertices, normales, caras, y por ultimo preparamos el modelo.
    bool loadOBJ(const QString& filename) {
        QFile file(filename);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            return false;
        }

        // Por si se necesita limpiar algun modelo anterior
        vertices.clear();
        objNormals.clear();
        computedNormals.clear();
        faces.clear();
        hasOBJNormals = false;

        // Reinicia los ,imites del modelo
        minBound = Vec3( 1e9f,  1e9f,  1e9f);
        maxBound = Vec3(-1e9f, -1e9f, -1e9f);

        QTextStream in(&file);
        while (!in.atEnd()) {
            QString line = in.readLine().trimmed();
            if (line.isEmpty() || line.startsWith('#')) continue;

            QStringList parts = line.split(' ', Qt::SkipEmptyParts);
            if (parts.isEmpty()) continue;

            // Genera la linea del vertice
            if (parts[0] == "v" && parts.size() >= 4) {
                Vec3 p(parts[1].toFloat(), parts[2].toFloat(), parts[3].toFloat());
                vertices.push_back(p);

                minBound.x = std::min(minBound.x, p.x);
                minBound.y = std::min(minBound.y, p.y);
                minBound.z = std::min(minBound.z, p.z);
                maxBound.x = std::max(maxBound.x, p.x);
                maxBound.y = std::max(maxBound.y, p.y);
                maxBound.z = std::max(maxBound.z, p.z);
            }
            // Línea de normal del archivo OBJ
            else if (parts[0] == "vn" && parts.size() >= 4) {
                Vec3 n(parts[1].toFloat(), parts[2].toFloat(), parts[3].toFloat());
                objNormals.push_back(normalize(n));
            }
            // Línea de cara
            else if (parts[0] == "f" && parts.size() >= 4) {
                std::vector<FaceVertex> idx;
                idx.reserve(parts.size() - 1);
                for (int i = 1; i < parts.size(); ++i) {
                    idx.push_back(parseFaceVertexToken(parts[i]));
                }

                // Cuando llega un triangulo se guarda directo
                if (idx.size() == 3) {
                    faces.push_back({idx[0], idx[1], idx[2]});
                }
                //Cuando llega un quad se triangula
                else if (idx.size() == 4) {
                    faces.push_back({idx[0], idx[1], idx[2]});
                    faces.push_back({idx[0], idx[2], idx[3]});
                }
                // Cuando llega un polígono mayor lo triangulamos en un abanico
                else if (idx.size() > 4) {
                    for (size_t i = 1; i + 1 < idx.size(); ++i) {
                        faces.push_back({idx[0], idx[i], idx[i + 1]});
                    }
                }
            }
        }

        // Calculamos las normales suaves por si el OBJ si no tiene normales utilizables
        computeFallbackNormals();

        // Detecta si al menos algunas esquinas sí traen normal del archivo.
        hasOBJNormals = !objNormals.empty();
        for (const Face& f : faces) {
            if (f.a.vn >= 0 || f.b.vn >= 0 || f.c.vn >= 0) {
                hasOBJNormals = true;
                break;
            }
        }

        // Calculamos el centro y escala para que se normalice el modelo.
        modelCenter = (minBound + maxBound) * 0.5f;
        Vec3 size = maxBound - minBound;
        float maxDim = max3(std::fabs(size.x), std::fabs(size.y), std::fabs(size.z));
        normalizeScale = (maxDim > 1e-8f) ? (2.0f / maxDim) : 1.0f;

        update();
        return !vertices.empty() && !faces.empty();
    }

    // Carga una imagen como textura
    bool loadTextureImage(const QString& filename) {
        QImage img;
        if (!img.load(filename)) {
            textureLoaded = false;
            textureImage = QImage();
            update();
            return false;
        }

        // Convertimos a RGB32 para que la lectura de los pixeles sea mas simple.
        textureImage = img.convertToFormat(QImage::Format_RGB32);
        textureLoaded = true;
        update();
        return true;
    }

    // Cambia la rotacion en x desde el slider
    void setAngleX(int value) { angleX = value * 3.14159265f / 180.0f; update(); }

    // Cambia la rotacion el y desde el slider
    void setAngleY(int value) { angleY = value * 3.14159265f / 180.0f; update(); }

    // Cambia entre perspectiva y ortogonal
    void setPerspective(bool value) { usePerspective = value; update(); }

    // Activa o desactiva el alambre
    void setWireframe(bool value) { wireframe = value; update(); }

    // Activa o desactiva la rotacion automatica
    void setAutoRotate(bool value) { autoRotate = value; }

    // Cambia el material
    void setMaterialIndex(int idx) { materialIndex = idx; update(); }

    // Cambia la camara
    void setCameraIndex(int idx) { cameraIndex = idx; update(); }

    // Enciende o apaga la luz azul
    void setBlueLightEnabled(bool value) { blueLight.enabled = value; update(); }

    // Quita y agrega la textura de imagen
    void setTextureEnabled(bool value) { textureEnabled = value; update(); }

    // Cambia cuantas veces podemos poner la textura en el modelo
    void setTextureRepeatIndex(int idx) {
        if (idx == 0) textureRepeat = 1.0f;
        else if (idx == 1) textureRepeat = 2.0f;
        else textureRepeat = 3.0f;
        update();
    }

    // Puede cambiar la resolucion del modelo para que sea mas facil de renderizar
    void setRenderScaleIndex(int idx) {
        if (idx == 0) renderScale = 1.0f;
        else if (idx == 1) renderScale = 0.75f;
        else renderScale = 0.50f;
        update();
    }

    // Avanza la rotacion automatica si está activa
    void stepRotation() {
        if (autoRotate) {
            angleY += 0.015f;
            update();
        }
    }

protected:
    // teclas para elegir entre camaras, luz azul y vista de alambre.
    void keyPressEvent(QKeyEvent* event) override {
        if (event->key() == Qt::Key_1) {
            cameraIndex = 0;
            update();
        } else if (event->key() == Qt::Key_2) {
            cameraIndex = 1;
            update();
        } else if (event->key() == Qt::Key_3) {
            cameraIndex = 2;
            update();
        } else if (event->key() == Qt::Key_B) {
            blueLight.enabled = !blueLight.enabled;
            update();
        } else if (event->key() == Qt::Key_W) {
            wireframe = !wireframe;
            update();
        } else if (event->key() == Qt::Key_T) {
            textureEnabled = !textureEnabled;
            update();
        } else {
            QWidget::keyPressEvent(event);
        }
    }

    // Genera todo el render que se vera en el software
    void paintEvent(QPaintEvent*) override {
        if (vertices.empty() || faces.empty()) {
            QPainter p(this);
            p.fillRect(rect(), QColor(18, 18, 26));
            p.setPen(Qt::white);
            p.drawText(rect(), Qt::AlignCenter, "No se encontro el OBJ");
            return;
        }

        // Podemos bajar la resolucion del render para mejorar la velocidad
        const int renderW = std::max(320, int(width() * renderScale));
        const int renderH = std::max(240, int(height() * renderScale));

        // Framebuffer interno
        QImage img(renderW, renderH, QImage::Format_RGB32);
        img.fill(QColor(18, 18, 26));

        // Z-buffer para saber qué pixel está más cerca
        std::vector<float> zbuf(renderW * renderH, std::numeric_limits<float>::infinity());

        // Elige el material y la camara que el usuario queira
        CameraData cam = cameras[cameraIndex];
        Material mat = materials[materialIndex];

        // Generamos una funcion auxiliar para las barycentricas y prueba de interior del triangulo
        auto edge = [](float ax, float ay, float bx, float by, float px, float py) {
            return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
        };

        // =========================
        // MATRICES DEL PIPELINE 4x4
        // =========================

        // Normalizamos el modelo, primero lo centralizamos y lo escalamos
        Mat4 normalizeTranslate = translation(-modelCenter.x, -modelCenter.y, -modelCenter.z);
        Mat4 normalizeScaleMat = scaling(normalizeScale, normalizeScale, normalizeScale);
        Mat4 normalizeModel = multiplyMat4(normalizeScaleMat, normalizeTranslate);

        // Transformaciones del usuario para la rotacion
        Mat4 rotX = rotationX(angleX);
        Mat4 rotY = rotationY(angleY);
        Mat4 rotZ = rotationZ(0.0f);

        // Como se genera el modelo final
        Mat4 model = multiplyMat4(rotY, multiplyMat4(rotX, multiplyMat4(rotZ, normalizeModel)));

        // Vista de la camara tipo lookAt
        Vec3 center(0.0f, 0.0f, 0.0f);
        Vec3 up(0.0f, 1.0f, 0.0f);
        Mat4 view = lookAt(cam.eye, center, up);

        // Proyección elegida por el usuario
        float aspect = (renderH > 0) ? float(renderW) / float(renderH) : 1.0f;
        Mat4 projection = usePerspective
            ? perspective(55.0f * 3.14159265f / 180.0f, aspect, 0.10f, 50.0f)
            : orthographic(-1.8f * aspect, 1.8f * aspect, -1.8f, 1.8f, 0.10f, 50.0f);

        // Viewport para pasar de NDC a pixeles de la imagen
        Mat4 viewportMat = viewport(float(renderW), float(renderH));

        // Combinaciones utiles de matrices
        Mat4 viewModel = multiplyMat4(view, model);
        Mat4 mvp = multiplyMat4(projection, viewModel);

        // Estructura con datos preparados para rasterizacion
        struct ScreenV {
            float x, y;
            float depth;
            float invW;
            Vec3 objOverW;
            Vec3 worldOverW;
            Vec3 normalOverW;
        };

        // Tomamos una imagen 2D y la aplicamos con un mapeo esférico
        auto sampleBaseColor = [&](const Vec3& PobjNorm, const Vec3& Pworld) -> Color3 {
            // Si el usuario quita la textura o la imagen no existe, pone un degradado neutro
            if (!textureEnabled || !textureLoaded || textureImage.isNull()) {
                float t = clamp01((Pworld.y + 1.2f) / 2.4f);
                Color3 low(0.78f, 0.78f, 0.82f);
                Color3 high(0.90f, 0.88f, 0.86f);
                return Color3(
                    low.r * (1.0f - t) + high.r * t,
                    low.g * (1.0f - t) + high.g * t,
                    low.b * (1.0f - t) + high.b * t
                );
            }
            Vec3 dir = normalize(PobjNorm);
            float ySafe = std::max(-1.0f, std::min(1.0f, dir.y));
            float u = 0.5f + std::atan2(dir.z, dir.x) / (2.0f * 3.14159265f);
            float v = 0.5f - std::asin(ySafe) / 3.14159265f;
            u *= textureRepeat;
            u = u - std::floor(u);
            v = clamp01(v);
            int tx = std::clamp(int(u * float(textureImage.width() - 1)), 0, textureImage.width() - 1);
            int ty = std::clamp(int(v * float(textureImage.height() - 1)), 0, textureImage.height() - 1);

            QColor qc = textureImage.pixelColor(tx, ty);
            return Color3(qc.red() / 255.0f, qc.green() / 255.0f, qc.blue() / 255.0f);
        };

        auto shadePixel = [&](const Vec3& Pworld, const Vec3& Nworld, const Color3& base) -> Color3 {
            Vec3 normal = normalize(Nworld);
            Vec3 V = normalize(cam.eye - Pworld);

            Color3 result(
                mat.ka.r * base.r,
                mat.ka.g * base.g,
                mat.ka.b * base.b
            );

            auto applyLight = [&](const LightData& light) -> Color3 {
                if (!light.enabled) return Color3(0, 0, 0);

                Vec3 L = normalize(light.pos - Pworld);
                float ndotl = std::max(0.0f, dot(normal, L));

                Color3 diffuse(
                    base.r * mat.kd.r * light.intensity.r * ndotl,
                    base.g * mat.kd.g * light.intensity.g * ndotl,
                    base.b * mat.kd.b * light.intensity.b * ndotl
                );

                Vec3 H = normalize(L + V);
                float ndoth = std::max(0.0f, dot(normal, H));
                float spec = std::pow(ndoth, mat.shininess);

                Color3 specular(
                    mat.ks.r * light.intensity.r * spec,
                    mat.ks.g * light.intensity.g * spec,
                    mat.ks.b * light.intensity.b * spec
                );

                return diffuse + specular;
            };

            result = result + applyLight(whiteLight);
            result = result + applyLight(blueLight);
            return clampColor(result);
        };

        // Ponemos un vertice del OBJ hasta la pantalla usando modelo-vista-proyeccion
        auto projectVertex = [&](const Vec3& pObj, const Vec3& nObj) -> ScreenV {
            Vec4 obj4(pObj.x, pObj.y, pObj.z, 1.0f);
            Vec4 objNorm4 = multiplyVec4(normalizeModel, obj4);
            Vec4 normal4(nObj.x, nObj.y, nObj.z, 0.0f);
            Vec4 world4 = multiplyVec4(model, obj4);
            Vec4 clip4 = multiplyVec4(mvp, obj4);
            Vec4 normalWorld4 = multiplyVec4(model, normal4);
            float safeW = (std::fabs(clip4.w) < 1e-6f)
                ? ((clip4.w >= 0.0f) ? 1e-6f : -1e-6f)
                : clip4.w;
            float invW = 1.0f / safeW;
            Vec4 ndc(clip4.x * invW, clip4.y * invW, clip4.z * invW, 1.0f);
            Vec4 screen4 = multiplyVec4(viewportMat, ndc);

            return {
                screen4.x,
                screen4.y,
                ndc.z,
                invW,
                xyz(objNorm4) * invW,
                xyz(world4) * invW,
                normalize(xyz(normalWorld4)) * invW
            };
        };

        // =========================
        // RASTERIZACIÓN TRIÁNGULO POR TRIÁNGULO
        // =========================

        for (const Face& f : faces) {
            // Leemos los tres vertices del triangulo en espacio objeto
            Vec3 p0Obj = vertices[f.a.v];
            Vec3 p1Obj = vertices[f.b.v];
            Vec3 p2Obj = vertices[f.c.v];

            // Tomamos las normales del OBJ o las de respaldo calculadas
            Vec3 n0Obj = getNormalForCorner(f.a);
            Vec3 n1Obj = getNormalForCorner(f.b);
            Vec3 n2Obj = getNormalForCorner(f.c);

            //  Transoformamos los tres puntos a mundo para culling y luz
            Vec3 p0World = xyz(multiplyVec4(model, Vec4(p0Obj.x, p0Obj.y, p0Obj.z, 1.0f)));
            Vec3 p1World = xyz(multiplyVec4(model, Vec4(p1Obj.x, p1Obj.y, p1Obj.z, 1.0f)));
            Vec3 p2World = xyz(multiplyVec4(model, Vec4(p2Obj.x, p2Obj.y, p2Obj.z, 1.0f)));

            // Generamos la normal plana de la cara en el mundo
            Vec3 faceNormalWorld = normalize(cross(p1World - p0World, p2World - p0World));

            // Hacemos la Backface culling para no rasterizar lo que no mira a la camara
            Vec3 viewDir = normalize(cam.eye - p0World);
            if (dot(faceNormalWorld, viewDir) <= 0.0f) continue;

            // Hacemos la proyeccion completa de los tres vertices
            ScreenV s0 = projectVertex(p0Obj, n0Obj);
            ScreenV s1 = projectVertex(p1Obj, n1Obj);
            ScreenV s2 = projectVertex(p2Obj, n2Obj);

            float area = edge(s0.x, s0.y, s1.x, s1.y, s2.x, s2.y);
            if (std::fabs(area) < 1e-8f) continue;
            int minX = std::max(0, (int)std::floor(std::min({s0.x, s1.x, s2.x})));
            int maxX = std::min(renderW - 1, (int)std::ceil(std::max({s0.x, s1.x, s2.x})));
            int minY = std::max(0, (int)std::floor(std::min({s0.y, s1.y, s2.y})));
            int maxY = std::min(renderH - 1, (int)std::ceil(std::max({s0.y, s1.y, s2.y})));
            for (int y = minY; y <= maxY; ++y) {
                for (int x = minX; x <= maxX; ++x) {
                    float px = x + 0.5f;
                    float py = y + 0.5f;
                    float w0 = edge(s1.x, s1.y, s2.x, s2.y, px, py);
                    float w1 = edge(s2.x, s2.y, s0.x, s0.y, px, py);
                    float w2 = edge(s0.x, s0.y, s1.x, s1.y, px, py);
                    bool inside = (area > 0.0f)
                        ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f)
                        : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
                    if (!inside) continue;
                    w0 /= area;
                    w1 /= area;
                    w2 /= area;
                    float interpInvW = s0.invW * w0 + s1.invW * w1 + s2.invW * w2;
                    if (std::fabs(interpInvW) < 1e-8f) continue;
                    float depth = usePerspective
                        ? (s0.depth * s0.invW * w0 + s1.depth * s1.invW * w1 + s2.depth * s2.invW * w2) / interpInvW
                        : (s0.depth * w0 + s1.depth * w1 + s2.depth * w2);
                    int idx = y * renderW + x;
                    if (depth < zbuf[idx]) {
                        zbuf[idx] = depth;
                        Vec3 PobjNorm = (s0.objOverW * w0 + s1.objOverW * w1 + s2.objOverW * w2) / interpInvW;
                        Vec3 Pworld = (s0.worldOverW * w0 + s1.worldOverW * w1 + s2.worldOverW * w2) / interpInvW;
                        Vec3 Nworld = normalize((s0.normalOverW * w0 + s1.normalOverW * w1 + s2.normalOverW * w2) / interpInvW);
                        Color3 base = sampleBaseColor(PobjNorm, Pworld);
                        Color3 c = shadePixel(Pworld, Nworld, base);
                        img.setPixelColor(x, y, QColor(
                            int(clamp01(c.r) * 255.0f),
                            int(clamp01(c.g) * 255.0f),
                            int(clamp01(c.b) * 255.0f)
                        ));
                    }
                }
            }

            // Genera el dibujo del alambre si el usuario lo activa
            if (wireframe) {
                QPainter ip(&img);
                ip.setPen(QPen(QColor(240, 240, 240), 1));
                ip.drawLine(QPointF(s0.x, s0.y), QPointF(s1.x, s1.y));
                ip.drawLine(QPointF(s1.x, s1.y), QPointF(s2.x, s2.y));
                ip.drawLine(QPointF(s2.x, s2.y), QPointF(s0.x, s0.y));
            }
        }

        // Pintamos la imagen interna escalada a un tamaño real del widget
        QPainter painter(this);
        painter.fillRect(rect(), QColor(18, 18, 26));
        painter.drawImage(rect(), img);
        painter.setPen(Qt::white);
        painter.drawText(10, 20, QString("OBJ con textura de imagen | Camaras: 1,2,3 | Luz azul: B | Textura: T | Alambre: W"));
        painter.drawText(10, 40, QString("Vertices: %1 | Triangulos: %2 | Normales OBJ: %3 | Escala interna: %4%% | Textura imagen: %5 | Repeticion U: %6x")
                                   .arg(vertices.size())
                                   .arg(faces.size())
                                   .arg(hasOBJNormals ? "Si" : "No, usando normales calculadas")
                                   .arg(int(renderScale * 100.0f))
                                   .arg((textureEnabled && textureLoaded) ? "Activa" : (textureLoaded ? "Apagada" : "No cargada"))
                                   .arg(textureRepeat, 0, 'f', 1));
    }

private:
    int parseOBJIndex(const QString& s, int currentCount) const {
        int idx = s.toInt();
        if (idx > 0) return idx - 1;
        if (idx < 0) return currentCount + idx;
        return -1;
    }

    FaceVertex parseFaceVertexToken(const QString& token) const {
        FaceVertex fv;
        QStringList sub = token.split('/');

        if (sub.size() >= 1 && !sub[0].isEmpty()) {
            fv.v = parseOBJIndex(sub[0], (int)vertices.size());
        }
        if (sub.size() >= 3 && !sub[2].isEmpty()) {
            fv.vn = parseOBJIndex(sub[2], (int)objNormals.size());
        }
        return fv;
    }

    void computeFallbackNormals() {
        computedNormals.assign(vertices.size(), Vec3(0, 0, 0));

        for (const Face& f : faces) {
            if (f.a.v < 0 || f.b.v < 0 || f.c.v < 0) continue;

            Vec3 p0 = vertices[f.a.v];
            Vec3 p1 = vertices[f.b.v];
            Vec3 p2 = vertices[f.c.v];
            Vec3 fn = normalize(cross(p1 - p0, p2 - p0));

            computedNormals[f.a.v] = computedNormals[f.a.v] + fn;
            computedNormals[f.b.v] = computedNormals[f.b.v] + fn;
            computedNormals[f.c.v] = computedNormals[f.c.v] + fn;
        }

        for (Vec3& n : computedNormals) n = normalize(n);
    }

    Vec3 getNormalForCorner(const FaceVertex& fv) const {
        if (fv.vn >= 0 && fv.vn < (int)objNormals.size()) {
            return objNormals[fv.vn];
        }
        if (fv.v >= 0 && fv.v < (int)computedNormals.size()) {
            return computedNormals[fv.v];
        }
        return Vec3(0, 1, 0);
    }

    // Geometria del modelo
    std::vector<Vec3> vertices;
    std::vector<Vec3> objNormals;
    std::vector<Vec3> computedNormals;
    std::vector<Face> faces;

    // Camaras y materiales
    std::vector<CameraData> cameras;
    std::vector<Material> materials;

    // Luces
    LightData whiteLight;
    LightData blueLight;

    // Información del modelo para normalizarlo
    Vec3 minBound;
    Vec3 maxBound;
    Vec3 modelCenter;
    float normalizeScale = 1.0f;
    bool hasOBJNormals = false;

    // Estado actual de la escena
    float angleX = -0.25f;
    float angleY = 0.75f;
    bool usePerspective = true;
    bool wireframe = false;
    bool autoRotate = false;
    bool textureEnabled = true;
    bool textureLoaded = false;
    QImage textureImage;
    int materialIndex = 0;
    int cameraIndex = 0;
    float textureRepeat = 1.0f;
    float renderScale = 0.50f;
};

// =========================
// VENTANA PRINCIPAL CON CONTROLES QT
// =========================

class MainWindow : public QMainWindow {
public:
    MainWindow() {
        // Contenedor y layout principal
        auto* central = new QWidget;
        auto* mainLayout = new QHBoxLayout(central);

        // Área donde se dibuja el render
        render = new RenderWidget;

        // Panel lateral con controles.
        auto* panel = new QWidget;
        auto* panelLayout = new QVBoxLayout(panel);

        // Slider de la rotacion x
        auto* lblX = new QLabel("Rotacion X");
        auto* sliderX = new QSlider(Qt::Horizontal);
        sliderX->setRange(-180, 180);
        sliderX->setValue(-14);

        // Slider de la rotacion y
        auto* lblY = new QLabel("Rotacion Y");
        auto* sliderY = new QSlider(Qt::Horizontal);
        sliderY->setRange(-180, 180);
        sliderY->setValue(43);

        // Selector de proyeccion
        auto* projLabel = new QLabel("Proyeccion");
        auto* projectionBox = new QComboBox;
        projectionBox->addItem("Perspectiva");
        projectionBox->addItem("Ortogonal");

        // Selector del material
        auto* materialLabel = new QLabel("Material");
        auto* materialBox = new QComboBox;
        materialBox->addItem("Material A");
        materialBox->addItem("Material B");

        // Selector de la camara
        auto* cameraLabel = new QLabel("Camara");
        auto* cameraBox = new QComboBox;
        cameraBox->addItem("Camara 1");
        cameraBox->addItem("Camara 2");
        cameraBox->addItem("Camara 3");

        // Selector de la resolucion
        auto* scaleLabel = new QLabel("Resolucion interna");
        auto* scaleBox = new QComboBox;
        scaleBox->addItem("100%");
        scaleBox->addItem("75%");
        scaleBox->addItem("50%");
        scaleBox->setCurrentIndex(2);

        // Selector de cuantas veces se repite la textura de imagen.
        auto* textureInfoLabel = new QLabel("Textura de imagen");
        auto* textureNameLabel = new QLabel("Archivo: textura_imagen.png");
        auto* repeatLabel = new QLabel("Repeticion horizontal");
        auto* repeatBox = new QComboBox;
        repeatBox->addItem("1x");
        repeatBox->addItem("2x");
        repeatBox->addItem("3x");

        // Casillas para seleccionar opciones variadas
        auto* wireframeCheck = new QCheckBox("Mostrar alambre");
        auto* autoRotateCheck = new QCheckBox("Rotacion automatica");
        auto* blueLightCheck = new QCheckBox("Luz azul activa");
        auto* textureCheck = new QCheckBox("Textura de imagen activa");
        blueLightCheck->setChecked(true);
        textureCheck->setChecked(true);

        // Botón para recargar el OBJ
        auto* reloadButton = new QPushButton("Recargar OBj y textura");

        // Agregamos controles al panel
        panelLayout->addWidget(lblX);
        panelLayout->addWidget(sliderX);
        panelLayout->addWidget(lblY);
        panelLayout->addWidget(sliderY);
        panelLayout->addWidget(projLabel);
        panelLayout->addWidget(projectionBox);
        panelLayout->addWidget(materialLabel);
        panelLayout->addWidget(materialBox);
        panelLayout->addWidget(cameraLabel);
        panelLayout->addWidget(cameraBox);
        panelLayout->addWidget(scaleLabel);
        panelLayout->addWidget(scaleBox);
        panelLayout->addWidget(textureInfoLabel);
        panelLayout->addWidget(textureNameLabel);
        panelLayout->addWidget(repeatLabel);
        panelLayout->addWidget(repeatBox);
        panelLayout->addWidget(wireframeCheck);
        panelLayout->addWidget(autoRotateCheck);
        panelLayout->addWidget(blueLightCheck);
        panelLayout->addWidget(textureCheck);
        panelLayout->addWidget(reloadButton);
        panelLayout->addStretch();

        // Ponemos el render y panel en la ventana principal
        mainLayout->addWidget(render, 1);
        mainLayout->addWidget(panel);

        setCentralWidget(central);
        setWindowTitle("Qt Proyecto OBJ");
        resize(1280, 780);

        // Conexiones de controles con el render
        QObject::connect(sliderX, &QSlider::valueChanged, render, &RenderWidget::setAngleX);
        QObject::connect(sliderY, &QSlider::valueChanged, render, &RenderWidget::setAngleY);
        QObject::connect(projectionBox, &QComboBox::currentIndexChanged, this, [=](int idx) {
            render->setPerspective(idx == 0);
        });
        QObject::connect(materialBox, &QComboBox::currentIndexChanged, render, [=](int idx) {
            render->setMaterialIndex(idx);
        });
        QObject::connect(cameraBox, &QComboBox::currentIndexChanged, render, &RenderWidget::setCameraIndex);
        QObject::connect(scaleBox, &QComboBox::currentIndexChanged, render, &RenderWidget::setRenderScaleIndex);
        QObject::connect(repeatBox, &QComboBox::currentIndexChanged, render, &RenderWidget::setTextureRepeatIndex);
        QObject::connect(wireframeCheck, &QCheckBox::toggled, render, &RenderWidget::setWireframe);
        QObject::connect(autoRotateCheck, &QCheckBox::toggled, render, &RenderWidget::setAutoRotate);
        QObject::connect(blueLightCheck, &QCheckBox::toggled, render, &RenderWidget::setBlueLightEnabled);
        QObject::connect(textureCheck, &QCheckBox::toggled, render, &RenderWidget::setTextureEnabled);

        // Generamos un boton para volver a cargar el OBJ y su imagen desde la carpeta del ejecutable
        QObject::connect(reloadButton, &QPushButton::clicked, this, [=]() {
            QString base = QCoreApplication::applicationDirPath();
            QString objPath = base + "/bunny.obj";
            QString texPath = base + "/textura_cafe.png";
            if (!render->loadOBJ(objPath)) {
                QMessageBox::warning(this, "Error", "No se pudo recargar bunny.obj");
            }
            if (!render->loadTextureImage(texPath)) {
                QMessageBox::warning(this, "Warning", "No se encontro textura_cafe.png junto al ejecutable");
            }
        });

        // Timer para la animacion
        timer = new QTimer(this);
        QObject::connect(timer, &QTimer::timeout, render, &RenderWidget::stepRotation);
        timer->start(16);

        // Cargamos la el OBJ y la imagen de textura.
        QString basePath = QCoreApplication::applicationDirPath();
        QString objPath = basePath + "/bunny.obj";
        QString texPath = basePath + "/textura_cafe.png";
        if (!render->loadOBJ(objPath)) {
            QMessageBox::warning(this, "Error", "No se encontro bunny.obj junto al ejecutable.");
        }
        if (!render->loadTextureImage(texPath)) {
            QMessageBox::warning(this, "Warning", "No se encontro textura_cafe.png junto al ejecutable");
        }
    }

private:
    RenderWidget* render = nullptr;
    QTimer* timer = nullptr;
};

// =========================
// FUNCIÓN PRINCIPAL
// =========================

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}
