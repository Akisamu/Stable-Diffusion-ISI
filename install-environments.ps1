$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
function InstallFail {
    Write-Output "InstallFail"
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

if (!(Test-Path -Path "venv")) {
    Write-Output "Creating venv"
    python -m venv venv
    Check "Unable to create venv"
}

.\venv\Scripts\activate
Check "venv install FAILURE"

if(Test-Path -Path "stable-diffusion") {
    Move-Item -Path ".\stable-diffusion\configs" -Destination ".\" -Recurse -Force
    Move-Item -Path ".\stable-diffusion\data" -Destination ".\" -Recurse -Force
    Move-Item -Path ".\stable-diffusion\ldm" -Destination ".\" -Recurse -Force
    Move-Item -Path ".\stable-diffusion\models" -Destination ".\" -Force
    Move-Item -Path ".\stable-diffusion\setup.py" -Destination ".\" -Force
    Remove-Item -Path ".\stable-diffusion\" -Recurse -Force
}

Write-Output "Install dependences."
pip3 install clean-fid numba numpy torch==2.0.0+cu118 torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu118
Check "Unable to  install PyTorch"
pip install --upgrade -r ./install/requirements.txt
Check "Unable to install dependences in requirements.txt"

Write-Output "Install venv success."
Read-Host | Out-Null ;
