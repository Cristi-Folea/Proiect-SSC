Proiectul P3

# Roadmap P3: Accelerarea Inferenței cu TensorRT (5 Săptămâni)

---

## Săptămâna 1: Setup și Export Model (PyTorch ➔ ONNX)

- [ ] **Setup mediu:** Configurarea mediului de dezvoltare pe Jetson Orin Nano (JetPack, TensorRT, CUDA Toolkit, OpenCV).
- [ ] **Alegere model:** Descărcarea unui model preantrenat optimizat pentru detecție în timp real (ex. **YOLOv8 Nano** sau **MobileNet**).
- [ ] **Validare PyTorch:** Rularea unei inferențe simple pe un set de imagini direct în PyTorch pentru verificarea acurateței inițiale.
- [ ] **Export ONNX:** Exportarea arhitecturii și a ponderilor din PyTorch în formatul intermediar `.onnx`.

---

## Săptămâna 2: Conversia și Optimizarea TensorRT

- [ ] **Generare Engine FP32:** Compilarea primului binar TensorRT în precizie completă **FP32** folosind utilitarul `trtexec` sau API-ul de C++/Python.
- [ ] **Generare Engine FP16:** Compilarea celei de-a doua variante binare în **FP16**, activând nucleele Tensor de pe Jetson Orin Nano.
- [ ] **Testare izolată:** Măsurarea latenței brute a fiecărui engine folosind date dummy (fictive) pentru a valida optimizarea.

---

## Săptămâna 3: Kerneluri CUDA Proprii (Pre/Postprocesare)

- [ ] **Preprocesare CUDA:** Scrierea unui kernel CUDA propriu în C++ pe GPU pentru:
  - Conversia spațiului de culoare (BGR la RGB).
  - Redimensionarea imaginii (resize).
  - Normalizarea pixelilor.
- [ ] **Postprocesare CUDA / C++:** Implementarea parsării vectorilor de ieșire și scrierea algoritmului **Non-Maximum Suppression (NMS)** pentru eliminarea casetelor suprapuse.
- [ ] **Optimizare memorie GPU:** Folosirea memoriei de tip _pinned_ sau _zero-copy_ (`cudaHostAlloc`) pentru transferuri de date fără gâturi de sticlă.

---

## Săptămâna 4: Live Pipeline & Feature de Nota 10

- [ ] **Captură live:** Conectarea camerei (Webcam USB / MIPI-CSI) și citirea cadrelor în timp real într-o aplicație C++ sau Python.
- [ ] **Asamblare pipeline:** Unificarea fluxului într-o buclă neblocantă:  
       `Captură Cadru` ➔ `Preprocesare CUDA` ➔ `Inference Engine TensorRT` ➔ `Postprocesare` ➔ `Afișare`
- [ ] **Feature de Nota 10 (Virtual Tripwire & HUD Analitice):**
  - Trasarea unei linii virtuale roșii pe imagine.
  - Implementarea logicii de urmărire a casetelor care trec peste linie și un contor de obiecte.
  - Proiectarea unui panou HUD live pe ecran cu: **FPS**, **Latență (ms)**, **Consum memorie GPU**.

---

## Săptămâna 5: Benchmarking, Switch Dinamic & Pregătire Demo

- [ ] **Switch Dinamic la Runtime:** Implementarea unei comenzi de la tastatură (ex. tasta `M`) pentru a comuta instant între engine-ul **FP32** și **FP16** în timpul rulării live.
- [ ] **Colectare Metrice:** Înregistrarea și compararea parametrilor ceruți în evaluare:
  - Latență și FPS (FP32 vs FP16).
  - Consum memorie GPU.
  - Speedup și analiza vizuală a acurateței.
- [ ] **Pregătire Demo:** Verificarea stabilității aplicației și pregătirea scenariului de prezentare live.
