# Used Car Price Predictor — Full-Stack App

## Arhitektura

```
Avalonia Desktop  ──HTTP──▶  ASP.NET Core 8 API  ──HTTP──▶  Python FastAPI  ──▶  best_model.pkl
  (Frontend)                     (Backend)                  (ML Service)
  Port: -                      Port: 5000                   Port: 8000
```

---

## Što trebaš instalirati

| Alat | Verzija | Link |
|------|---------|------|
| .NET SDK | 8.0+ | https://dotnet.microsoft.com/download |
| Python | 3.10+ | https://www.python.org/downloads/ |
| pip | (dolazi s Pythonom) | - |

---

## Koraci – jedanput (setup)

### 1. Pripremi ML artefakte (Python)

Ovi fajlovi MORAJU postojati iz tvojih Python skripti:

```
models/best_model.pkl      ← generirano s train.py
data/preprocessor.pkl      ← generirano s preprocessing.py
data/X_train_proc.npy      ← generirano s preprocessing.py
data/y_train.npy           ← generirano s preprocessing.py
```

Pokud već imaš te fajlove, idi na korak 2.

**Ako nisi još trenirao model:**
```bash
python preprocessing.py    # kreira data/
python train.py            # kreira models/best_model.pkl
```

### 2. Izvezi Target Encoding mape

```bash
cd MLService
python export_te_maps.py
# → kreira data/te_maps.pkl
```

Ovo je potrebno samo jedanput (ili ako ponovo trenirate model).

### 3. Instaliraj Python pakete

```bash
cd MLService
pip install -r requirements.txt
```

---

## Pokretanje aplikacije (3 terminala)

### Terminal 1 – Python ML Service

```bash
cd MLService
uvicorn main:app --host 0.0.0.0 --port 8000
```

Provjeri: http://localhost:8000/health → `{"status":"ok","model_loaded":true}`

---

### Terminal 2 – ASP.NET Core Backend

```bash
cd Backend/UsedCarsApi
dotnet run
```

Provjeri: http://localhost:5000/api/cars/health  
Swagger UI: http://localhost:5000/swagger

---

### Terminal 3 – Avalonia Frontend

```bash
cd Frontend/UsedCarsApp
dotnet run
```

Otvara se desktop prozor aplikacije.

---

## Struktura projekta

```
UsedCarsPricePrediction/
├── MLService/
│   ├── main.py              ← FastAPI wrapper oko sklearn modela
│   ├── export_te_maps.py    ← Jedanput pokreni za TE mape
│   └── requirements.txt
│
├── Backend/
│   └── UsedCarsApi/
│       ├── Controllers/
│       │   └── CarsController.cs
│       ├── Services/
│       │   └── CarPredictionService.cs
│       ├── Models/
│       │   └── CarModels.cs
│       ├── Program.cs
│       └── appsettings.json
│
├── Frontend/
│   └── UsedCarsApp/
│       ├── ViewModels/
│       │   ├── MainViewModel.cs     ← svi bindani podaci + komande
│       │   └── ViewModelLocator.cs
│       ├── Views/
│       │   ├── MainWindow.axaml     ← UI layout
│       │   └── MainWindow.axaml.cs
│       ├── Models/
│       │   └── CarModels.cs
│       ├── Services/
│       │   └── CarApiService.cs     ← HTTP pozivi prema backendu
│       ├── App.axaml
│       ├── App.axaml.cs
│       └── Program.cs
│
├── models/
│   └── best_model.pkl       ← tvoj trenirani model (NIJE u git-u)
├── data/
│   ├── preprocessor.pkl
│   ├── te_maps.pkl          ← generirano s export_te_maps.py
│   └── *.npy
└── README.md
```

---

## Konfiguracija portova

Ako trebaš promijeniti portove:

- **ML Service port**: promijeni u `uvicorn main:app --port XXXX` i u `Backend/appsettings.json` → `MlService:BaseUrl`
- **Backend port**: promijeni u `appsettings.json` → `Urls`
- **Frontend target**: promijeni u `Frontend/UsedCarsApp/Views/MainWindow.axaml.cs` → `new CarApiService("http://localhost:XXXX")`

---

## Česti problemi

| Problem | Rješenje |
|---------|----------|
| `model_loaded: false` | Provjeri putanje do `.pkl` fajlova u `main.py` |
| `502 Bad Gateway` | Python ML service nije pokrenut |
| Prazni dropdowni | Backend ne može dohvatiti `/api/cars/options` |
| `KeyError` u te_maps | Pokreni `export_te_maps.py` ponovo s istim `used_cars.csv` |
| Avalonia build error | Provjeri .NET 8 SDK: `dotnet --version` |