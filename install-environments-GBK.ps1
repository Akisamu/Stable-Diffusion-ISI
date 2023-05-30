$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
function InstallFail {
    Write-Output "安装失败。"
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
    Write-Output "正在创建虚拟环境..."
    python -m venv venv
    Check "创建虚拟环境失败，请检查 python 是否安装完毕以及 python 版本。"
}

.\venv\Scripts\activate
Check "激活虚拟环境失败。"

if(Test-Path -Path "stable-diffusion") {
    Move-Item -Path ".\stable-diffusion\configs" -Destination ".\" -Recurse -Force
    Move-Item -Path ".\stable-diffusion\data" -Destination ".\" -Recurse -Force
    Move-Item -Path ".\stable-diffusion\ldm" -Destination ".\" -Recurse -Force
    Move-Item -Path ".\stable-diffusion\models" -Destination ".\" -Recurse -Force 
    Move-Item -Path ".\stable-diffusion\setup.py" -Destination ".\" -Force
    Remove-Item -Path ".\stable-diffusion\" -Recurse -Force
}

Write-Output "安装程序所需依赖 (已进行国内加速，若无法使用加速源请用 install.ps1)..."
Set-Location .\sd-scripts
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple
Check "torch 安装失败，请删除 venv 文件夹后重新运行。"
pip install --upgrade -r ./install/requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple
Check "其他依赖安装失败。"

Write-Output "安装完毕"
Read-Host | Out-Null ;
