## Курсовая работа Video Super Resolution

### Результат:
  - Созданы 2 модели: SRGaN и SRResNet
  - Имплементирована потоковая обработка видеофайлов

### Пример запуска
```python main.py -data_path Data/Datasets -load AICV-Course-Work/Data/Models/model_srgan.pth -train true -test false -epoch_count 1 -batch_size 1```

При наличии видео в папке Data/Datasets/raw программа их обработает автоматически и запишет файлы в более высоком разрешении в директорию Data/Datasets/<название файла>
