# NucleiSegmentationProject
Segmentaion of nuclei with UNET


Описание тренировки и использования нейросети для сегментации клеточных ядер
1.	Тренировка нейросети
1)	Тренировка при наличии средств вычислительной техники с GPU
Windows 10, x64 
GPU: Nvidia GeForce GTX 1050.
Предполагается, что для видеокарты уже установлены драйвера.
Чтобы использовать ресурсы GPU следует установить cudatoolkit и cudnn. Проще всего это сделать с использованием платформы для разработки Anaconda: 
https://www.anaconda.com/products/distribution
После установки следует запустить командную строку Anaconda PowerShell (от имени администратора, если установка самой anaconda производилась от имени администратора), и выполнить в ней команду:
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
После этого требуется установить tensorflow. Для этого нужно перейти в командную строку Windows и оттуда выполнить:
python -m pip install "tensorflow<2.11"
Убедиться, что GPU найдено, можно следующим образом:
python -c "import tensorflow as tf; print('Number of GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"

При запуске тренировки GPU будет выбраться автоматически вместо CPU.

После тренировки нейросети возможно появление синего экрана с ошибкой nvlddmkm.sys system_thread_exception_not_handled (https://windows10i.ru/ustranenie-nepoladok/nvlddmkm-sys-sinij-ekran-windows-10.html). Если такое случилось, то можно попробовать откатить драйвер видеокарты (в диспетчере устройств зайти в свойства видеокарты, а потом откатить драйвер). Либо полностью переустановить драйвера на видеокарту (поставить более старой версии, например). Но для этого сначала нужно удалить текущую версию драйверов (включая драйвера на звук и PhysX) с помощью ПО 
display driver uninstaller
https://www.guru3d.com/files-details/display-driver-uninstaller-download.html

Примечание: после установки Anaconda следует вручную добавить пути в переменную path: C:\Anaconda3, C:\Anaconda3\Scripts, C:\Anaconda3\Library\bin. При условии, что Anaconda была установлена в каталог Anaconda3.


