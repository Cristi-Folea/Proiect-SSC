# Proiectul P3

# Roadmap P3: Detectie YOLOv8n accelerata cu TensorRT

## Scopul proiectului

Implementarea pe NVIDIA Jetson Orin Nano a unei aplicatii de detectie de obiecte folosind YOLOv8n, PyTorch, ONNX TensorRT, C++ si CUDA.

Aplicatia finala va:

- rula YOLOv8n in PyTorch pentru obtinerea unei referinte;
- exporta modelul in ONNX;
- construi engine-uri TensorRT FP32 si FP16;
- folosi preprocesare si postprocesare accelerate cu CUDA;
- procesa imagini si flux video;
- afisa bounding box-uri, clase si scoruri de incredere;
- urmari obiectele prin ID-uri;
- calcula directia de deplasare;
- detecta traversarea unei linii virtuale (tripwire);
- schimba culoarea bounding box-ului si contoriza obiectele care au trecut.

## Saptamana 1: Mediu configurat YOLOv8n in PyTorch si exportul ONNX

### Obiectiv

Alegerea modelului, configurarea mediului si obtinerea unei versiuni YOLOv8n care poate detecta obiecte in PyTorch si poate fi exportata in ONNX.

### Activitati

- Configurarea Jetson Orin Nano cu JetPack, CUDA, cuDNN, TensorRT, PyTorch,
  ONNX si OpenCV.
- Descarcarea modelului preantrenat YOLOv8n.
- Stabilirea intrarii modelului, de exemplu `640 x 640`, batch size `1`.
- Rularea inferentei PyTorch pe imagini si pe cel putin un videoclip scurt.
- Salvarea clasei, scorului de incredere si coordonatelor fiecarui bounding box.
- Exportarea modelului in ONNX.
- Validarea fisierului ONNX si compararea rezultatelor PyTorch cu ONNX.

### Livrabile in `/sources`

```text
sources/python/yolo_baseline.py
sources/python/export_yolov8_onnx.py
sources/python/validate_onnx.py
sources/tests/test_onnx.py
sources/configs/yolov8n.yaml
```

### Artefacte generate

```text
artifacts/models/yolov8n.onnx
results/yolov8n_baseline.json
results/onnx_validation.json
```

## Saptamana 2: Engine-uri TensorRT FP32 si FP16

### Obiectiv

Conversia aceluiasi model ONNX in doua engine-uri TensorRT si implementarea incarcarii si executiei lor din C++.

### Activitati

- Verificarea compatibilitatii YOLOv8n ONNX cu TensorRT.
- Construirea engine-ului `FP32`.
- Construirea engine-ului `FP16`.
- Folosirea aceleiasi forme de intrare pentru ambele engine-uri.
- Implementarea clasei C++ pentru:
  - incarcarea engine-ului;
  - identificarea tensorilor;
  - alocarea bufferelor;
  - lansarea inferentei;
  - raportarea explicita a erorilor TensorRT si CUDA.
- Rularea unei inferente de test cu fiecare engine.

### Livrabile in `/sources`

```text
sources/cpp/CMakeLists.txt
sources/cpp/include/trt_runner.hpp
sources/cpp/src/trt_runner.cpp
sources/scripts/build_engines.sh
```

### Surse generate

```text
artifacts/engines/yolov8n_fp32.engine
artifacts/engines/yolov8n_fp16.engine
```

## Saptamana 3: Preprocesare si postprocesare CUDA

### Obiectiv

Implementarea etapelor de preprocesare si postprocesare pe GPU, astfel incat pipeline-ul sa poata fi folosit de aplicatia TensorRT.

### Activitati

#### Preprocesare CUDA

- conversie BGR in RGB;
- resize sau letterbox la dimensiunea modelului;
- normalizarea pixelilor;
- copierea datelor in bufferul de intrare GPU.

#### Postprocesare CUDA

- interpretarea tensorului de iesire YOLOv8n;
- filtrarea dupa confidence threshold;
- transformarea coordonatelor in dimensiunile originale;
- Non-Maximum Suppression;
- returnarea clasei, scorului si coordonatelor fiecarui bounding box.

#### Testare

- Compararea rezultatului CUDA cu o implementare CPU de referinta.
- Verificarea aceleiasi clase pentru aceleasi imagini.
- Verificarea dimensiunilor si coordonatelor bounding box-urilor.
- Raportarea explicita a erorilor de sincronizare sau alocare CUDA.

### Livrabile in `/sources`

```text
sources/cpp/include/cuda_preprocess.cuh
sources/cpp/src/cuda_preprocess.cu
sources/cpp/include/yolo_postprocess.cuh
sources/cpp/src/yolo_postprocess.cu
sources/tests/test_cuda_pipeline.cpp
```

### Livrabil la sfarsitul saptamanii

Un pipeline functional:

```text
imagine -> preprocesare CUDA -> TensorRT -> postprocesare CUDA -> detectii
```

Exemplu de detectie:

```text
class_name: person
confidence: 0.91
box: x1, y1, x2, y2
```

---

## Saptamana 4: Aplicatie video, tracking si tripwire

### Obiectiv

Integrarea detectiei intr-o aplicatie video si implementarea feature-ului de tracking si tripwire.

### Activitati

- Citirea unui videoclip sau a unei camere USB/MIPI cu OpenCV.
- Integrarea pipeline-ului complet:

```text
captura video
    -> preprocesare CUDA
    -> TensorRT FP32 sau FP16
    -> postprocesare CUDA
    -> tracking
    -> tripwire
    -> afisare
```

- Atribuirea unui `track_id` persistent fiecarui obiect.
- Asocierea detectiei din cadrul curent cu detectia din cadrul anterior.
- Calcularea centrului bounding box-ului.
- Definirea unei linii virtuale prin doua puncte configurabile.
- Detectarea schimbarii partii liniei de catre centrul obiectului.
- Generarea unui singur eveniment pentru fiecare obiect care traverseaza linia.
- Schimbarea culorii bounding box-ului:
  - verde: obiect detectat;
  - galben: obiect urmarit;
  - rosu: obiectul a traversat tripwire-ul.
- Afisarea pe imagine a ID-ului si a clasei.
- Afisarea unui contor de obiecte traversate, grupat dupa clasa si directie.

### Livrabile in `/sources`

```text
sources/cpp/src/main.cpp
sources/cpp/include/tracker.hpp
sources/cpp/src/tracker.cpp
sources/cpp/include/tripwire.hpp
sources/cpp/src/tripwire.cpp
sources/cpp/README.md
```

## Saptamana 5: Benchmark, comparatie si demo final

### Obiectiv

Masurarea reproductibila a performantei si pregatirea demonstratiei finale.

### Activitati

- Rularea aceluiasi set de imagini sau videoclipuri pentru PyTorch, FP32 si
  FP16.
- Executarea unei perioade de warm-up inainte de masuratori.
- Realizarea a cel putin 100 de inferente masurate pentru fiecare configuratie.
- Masurarea:
  - latentei medii, mediane si P95;
  - FPS;
  - timpului de preprocesare;
  - timpului TensorRT;
  - timpului de postprocesare;
  - memoriei GPU;
  - numarului de detectii;
  - diferentelor dintre rezultatele FP32 si FP16.
- Calcularea accelerarii:

```text
speedup = latenta_FP32 / latenta_FP16
```

- Salvarea evenimentelor tripwire in format CSV.
- Verificarea ca un obiect este numarat o singura data.
- Pregatirea unei demonstratii cu un videoclip controlat.
- Documentarea comenzilor de compilare, rulare si benchmark.

### Livrabile in `/sources`

```text
sources/benchmark/benchmark.py
sources/scripts/run_benchmark.sh
sources/README.md
```

## Rezultatul final

La sfarsitul celor 5 saptamani trebuie sa existe:

1. model YOLOv8n rulat in PyTorch;
2. model exportat in ONNX;
3. engine TensorRT FP32;
4. engine TensorRT FP16;
5. aplicatie C++ pentru imagini si video;
6. preprocesare si postprocesare CUDA;
7. tracking cu ID-uri;
8. directie de deplasare;
9. tripwire functional;
10. bounding box colorat diferit dupa starea obiectului;
11. contorizarea obiectelor care au traversat;
12. benchmark reproductibil si raport final.
