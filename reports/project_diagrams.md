# MindSight Project Diagrams

These diagrams describe the workflow implemented by the repository as of August 2026.
MindSight is an assistive screening aid, not a diagnostic or clinical system.

## 1. End-to-End Process Diagram

The project has two connected lifecycles: an offline data/model lifecycle and an online screening lifecycle.

```mermaid
flowchart LR
    subgraph Offline[Offline preparation and model lifecycle]
        A[Raw Dreaddit CSV files] --> B[Load train and test data]
        B --> C[Inspect quality and Reddit artifacts]
        C --> D[Generate EDA figures and report]
        B --> E[Save processed CSV copies]
        E --> F[Feature and model development]
        F --> G[(models/ artifacts)]
    end

    subgraph Online[Online screening lifecycle]
        H[User text or uploaded CSV] --> I[Streamlit UI]
        I --> J{FastAPI reachable?}
        J -->|Yes| K[POST /predict]
        J -->|No| L[Local heuristic engine]
        K --> M[Clean text and infer score]
        L --> N[Clean text and infer score]
        G -. optional persisted artifacts .-> M
        M --> O[Apply threshold]
        N --> O
        O --> P[Risk label, confidence, features, explanations]
        P --> Q[Charts, trajectory or batch results]
        Q --> R[Display disclaimer and resources]
    end

    D --> S[(reports/)]
```

## 2. Single-Post Screening Flow Chart

```mermaid
flowchart TD
    A([Start]) --> B[Select model and threshold]
    B --> C[Enter text or choose preset]
    C --> D{Text contains valid content?}
    D -->|No| E[Show awaiting-input state]
    D -->|Yes| F[MindSightClient.predict]
    F --> G{API request succeeds?}
    G -->|No| H[Run local heuristic fallback]
    G -->|Yes| I[FastAPI /predict]
    I --> J[Clean HTML, URLs, Reddit mentions and deleted markers]
    J --> K{Model files available?}
    K -->|Yes| L[Load logistic regression and TF-IDF artifacts]
    K -->|No| M[Use linguistic signal heuristic]
    L --> N[Calculate confidence and explanations]
    M --> N
    H --> O[Calculate local confidence and explanations]
    N --> P[Compare confidence with threshold]
    O --> P
    P --> Q[Return stressed or not stressed]
    Q --> R[Render gauge, metrics, tokens and features]
    R --> S{Elevated distress?}
    S -->|Yes| T[Show crisis resources and disclaimer]
    S -->|No| U[Show disclaimer]
    T --> V([End])
    U --> V
    E --> V
```

## 3. Data Flow Diagrams

### 3.1 Context / Level 0 DFD

```mermaid
flowchart LR
    User[User or analyst] -->|Text, model choice, threshold, CSV| System((MindSight screening system))
    System -->|Risk result, confidence, features, explanations| User
    System -->|Health status| User
    Dataset[Dataset provider / local Dreaddit files] -->|Raw train and test CSV| System
    ModelStore[(Model artifact directory)] -->|Optional TF-IDF and classifier files| System
    System -->|EDA report and figures| Reports[(Reports directory)]
```

### 3.2 Level 1 DFD

```mermaid
flowchart TB
    User[User or analyst]
    Raw[(data/raw)]
    Processed[(data/processed)]
    Models[(models)]
    Reports[(reports)]
    UI[1. Streamlit interface]
    Client[2. API client and fallback]
    API[3. FastAPI service]
    Clean[4. Text cleaning]
    Signals[5. Linguistic signal extraction]
    Classifier[6. Optional persisted classifier inference]
    Decision[7. Threshold decision]
    Presentation[8. Result presentation]

    Raw -->|train/test CSV| Clean
    Clean -->|prepared text| Processed
    Raw -->|EDA input| Reports
    User -->|text, CSV, settings| UI
    UI --> Client
    Client -->|HTTP /predict| API
    Client -->|network failure| Signals
    API --> Clean
    Clean --> Signals
    Models --> Classifier
    Signals --> Classifier
    Signals --> Decision
    Classifier --> Decision
    Decision -->|risk, confidence, features, tokens| Presentation
    Presentation --> UI
    UI --> User
    Reports --> User
```

### 3.3 Main Data Stores and Contracts

| Store or interface | Contents | Written/read by |
|---|---|---|
| `data/raw/` | `dreaddit-train.csv`, `dreaddit-test.csv` | Dataset pipeline |
| `data/processed/` | Cleaned CSV copies | Dataset pipeline |
| `models/` | Optional `logistic_regression.pkl` and `tfidf_vectorizer.pkl` | FastAPI inference |
| `reports/` | EDA Markdown report and generated figures | Dataset pipeline, analysts |
| `POST /predict` | Text, model type, threshold to risk response | Streamlit client, FastAPI |
| `POST /explain` | Text and threshold to explanation response | API consumers |
| `GET /health` | Service health status | Streamlit sidebar, operators |

## 4. Gantt Chart

This is a practical eight-week delivery schedule. The first three activities match the currently visible repository capabilities; model training, validation, and production hardening remain work items because their implementation is not present in `src/models`.

```mermaid
gantt
    title MindSight implementation and delivery schedule
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Data foundation
    Dataset acquisition and DVC setup       :done, data1, 2026-08-24, 5d
    Data cleaning and artifact scan         :done, data2, after data1, 5d
    EDA figures and quality report          :done, data3, after data2, 5d

    section Modeling
    Feature engineering and baselines       :model1, after data3, 10d
    Transformer model experiments            :model2, after model1, 10d
    Evaluation, calibration and error audit :model3, after model2, 5d

    section Product integration
    FastAPI contracts and inference           :done, api1, 2026-09-14, 5d
    Streamlit single-post screening           :done, ui1, after api1, 5d
    Thread and batch workflows                :done, ui2, after ui1, 5d
    Explainability and ethics review          :ui3, after model3, 5d

    section Release
    Unit and integration tests                :test1, after ui3, 5d
    Docker deployment and smoke test         :deploy1, after test1, 5d
    Documentation and operational handoff   :docs1, after deploy1, 5d
```

## 5. Scope Notes

- `/predict` uses persisted model files only when both expected files exist; otherwise it uses the API heuristic.
- The Streamlit client uses its own equivalent heuristic when the API is unavailable.
- Batch CSV screening and thread trajectory analysis call the same prediction client repeatedly.
- The current application does not persist user-entered screening text according to the UI's stated privacy behavior.
- The risk label is a thresholded screening signal and must not be interpreted as a diagnosis.