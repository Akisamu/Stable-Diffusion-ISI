$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
function InstallFail {
    Write-Output "��װʧ�ܡ�"
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
    Write-Output "���ڴ������⻷��..."
    python -m venv venv
    Check "�������⻷��ʧ�ܣ����� python �Ƿ�װ����Լ� python �汾��"
}

.\venv\Scripts\activate
Check "�������⻷��ʧ�ܡ�"

if(Test-Path -Path "stable-diffusion") {
    Move-Item -Path ".\stable-diffusion\configs" -Destination ".\" -Recurse -Force
    Move-Item -Path ".\stable-diffusion\data" -Destination ".\" -Recurse -Force
    Move-Item -Path ".\stable-diffusion\ldm" -Destination ".\" -Recurse -Force
    Move-Item -Path ".\stable-diffusion\models" -Destination ".\" -Recurse -Force 
    Move-Item -Path ".\stable-diffusion\setup.py" -Destination ".\" -Force
    Remove-Item -Path ".\stable-diffusion\" -Recurse -Force
}

Write-Output "��װ������������ (�ѽ��й��ڼ��٣����޷�ʹ�ü���Դ���� install.ps1)..."
Set-Location .\sd-scripts
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirrors.bfsu.edu.cn/pypi/web/simple
Check "torch ��װʧ�ܣ���ɾ�� venv �ļ��к��������С�"
pip install --upgrade -r ./install/requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple
Check "����������װʧ�ܡ�"

Write-Output "��װ���"
Read-Host | Out-Null ;
