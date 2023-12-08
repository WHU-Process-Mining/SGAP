# SGAP

## Setup

```bash
conda env create -f env.yaml
```

## Generate kfold data file
```bash
conda python generate_dataset.py
```

## Preprocess input data
```bash
conda python generate_model_input.py
```

## Training with auto optimization

```bash
python run.py
```

## Testing

```bash
python test.py
```