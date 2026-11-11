# Proiect SSC P3

## Pe Jetson de rulat:

### 1. Verificare Mediu

``` bash
sudo apt update
sudo apt install -y git cmake build-essential
```

``` bash
nvcc --version
cmake --version
gcc --version
```

```bash
#Tensort:
dpkg -l | grep -E 'nvinfer|tensorrt'
which trtexec
```

### 2. Descarcare repo:
```bash
cd ~
git clone 
cd Proiect-SSC
```

### 3. Generarea engine-urilor TensorRT
```bash
chmod +x sources/scripts/build_engines.sh
./sources/scripts/build_engines.sh
ls -lh sources/engines/
```

### 4. Compilarea aplicatiei C++