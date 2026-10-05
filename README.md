# AutoPlot Civil

Civil engineering calculations and charts that run directly in the browser.

## Modules

| Module | Contents |
|---|---|
| Concrete | Cylinder compressive strength with h/d ratio correction (SNI 1974:2011) |
| Sieve | Grain-size distribution curve, Cu, Cc, USCS & AASHTO classification |
| Compaction | Proctor curve, MDD, OMC, zero air voids line |
| Statistics | Mean, standard deviation, characteristic strength, normality test |
| Units | Civil engineering unit conversion |

## Running

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Running tests

```bash
python -m pytest tests -q
```

## Example data

`examples/` contains a sample CSV file for each module.

## Structure

```
app.py                     Streamlit application
src/engineering/           calculation modules
src/visualization/         chart builders
tests/                     unit tests
examples/                  sample data
docs/                      formula summary
```

## License

MIT.