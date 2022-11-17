python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install -r requirements.txt
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --find-links https://download.pytorch.org/whl/torch_stable.html
python -m pip install git+https://github.com/geo-stack/detectron2.git@v0.6_win64_fix
python -m pip install spyder-kernels==2.3.*
pause
