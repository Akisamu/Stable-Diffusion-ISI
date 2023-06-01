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

Write-Output "Install dependences."
pip3 install clean-fid numba numpy torch==2.0.0+cu118 torchvision --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu118
Check "Unable to  install PyTorch"
pip install --upgrade -r ./install/requirements.txt
Check "Unable to install dependences in requirements.txt"

Write-Output "Install venv success."
Read-Host | Out-Null ;
