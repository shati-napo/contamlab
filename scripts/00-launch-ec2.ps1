<#
scripts/00-launch-ec2.ps1 — 手元(Windows)から GPU インスタンスを1台起こす。

    # まず見るだけ(既定。課金しない)
    .\scripts\00-launch-ec2.ps1 -KeyName my-key -SecurityGroupId sg-xxxx

    # 実際に起動する
    .\scripts\00-launch-ec2.ps1 -KeyName my-key -SecurityGroupId sg-xxxx -Execute

★ 既定は「組み立てたコマンドを表示するだけ」である。-Execute を付けるまで課金しない。
  contamlab の CLI が実 API に --yes を要求するのと同じ考えで、金がかかる操作を
  勢いで走らせないため。

★ IAM インスタンスプロファイルは**わざと付けない。**
  付けなければ、このインスタンスは AWS のマネージド推論 API を呼ぶ権限を
  そもそも持てない。CLAUDE.md の「絶対禁止」を、規律ではなく権限で担保する。
  (S3 も同様に不要。**非公開シードと HOLDOUT はインスタンス内に閉じる。**)
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$KeyName,
    [Parameter(Mandatory = $true)][string]$SecurityGroupId,
    [string]$Region = "ap-northeast-1",
    # L4 24GB。13B の Q4_K_M が約 8.4GB、モデルは harness が逐次評価するので
    # ピークは最大の1本ぶん。48GB(g6e.xlarge)は要らない。
    [string]$InstanceType = "g6.xlarge",
    [int]$VolumeGb = 200,
    # Spot でよい。応答キャッシュは追記専用なので、中断されても続きから再開できる。
    [switch]$Spot,
    [switch]$Execute
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    throw "aws CLI が見つからない。先に導入して 'aws configure' を済ませること。"
}

# --- AMI の解決 -------------------------------------------------------------
# ドライバ入りの Deep Learning AMI があればそれを使う(ドライバ導入の手間が消える)。
# 無ければ素の Ubuntu にして、10-bootstrap.sh 側でドライバを入れる。
$amiCandidates = @(
    "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id",
    "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id",
    "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
)

$amiId = $null
$amiParam = $null
foreach ($param in $amiCandidates) {
    try {
        $value = aws ssm get-parameter --region $Region --name $param `
            --query "Parameter.Value" --output text 2>$null
        if ($LASTEXITCODE -eq 0 -and $value -match '^ami-') {
            $amiId = $value.Trim(); $amiParam = $param; break
        }
    } catch { }
}
if (-not $amiId) {
    throw "AMI を解決できなかった。--image-id を手で指定すること。試した: $($amiCandidates -join ', ')"
}

Write-Host "AMI      : $amiId"
Write-Host "  由来   : $amiParam"
if ($amiParam -notmatch "deeplearning") {
    Write-Host "  ★ ドライバ無しの AMI。10-bootstrap.sh がドライバを入れて一度終了するので、" -ForegroundColor Yellow
    Write-Host "    再起動してから bootstrap を再実行すること。" -ForegroundColor Yellow
}

# --- 起動コマンドの組み立て -------------------------------------------------
$tagSpec = "ResourceType=instance,Tags=[{Key=Name,Value=contamlab},{Key=Project,Value=contamlab}]"
$blockDev = "DeviceName=/dev/sda1,Ebs={VolumeSize=$VolumeGb,VolumeType=gp3,DeleteOnTermination=true}"

$argsList = @(
    "ec2", "run-instances",
    "--region", $Region,
    "--image-id", $amiId,
    "--instance-type", $InstanceType,
    "--key-name", $KeyName,
    "--security-group-ids", $SecurityGroupId,
    "--block-device-mappings", $blockDev,
    "--tag-specifications", $tagSpec,
    # IMDSv2 必須。メタデータの取り出しにトークンを要求する(30-record-environment.sh は対応済み)。
    "--metadata-options", "HttpTokens=required,HttpEndpoint=enabled",
    "--count", "1"
)
if ($Spot) {
    $argsList += @("--instance-market-options", "MarketType=spot")
}

Write-Host ""
Write-Host "実行するコマンド:" -ForegroundColor Cyan
Write-Host ("  aws " + ($argsList -join " "))
Write-Host ""
Write-Host "課金の目安(東京・オンデマンド):" -ForegroundColor Cyan
Write-Host "  g6.xlarge  ≒ `$0.8/h    g6e.xlarge ≒ `$1.9/h    Spot なら概ね 1/3"
Write-Host "  gp3 ${VolumeGb}GB ≒ `$0.1/GB/月(起動していなくても掛かる)"
Write-Host ""

if (-not $Execute) {
    Write-Host "★ 表示しただけ。実際に起動するには -Execute を付ける。" -ForegroundColor Yellow
    exit 0
}

$json = & aws @argsList | ConvertFrom-Json
$instanceId = $json.Instances[0].InstanceId
Write-Host "起動した: $instanceId"

Write-Host "起動待ち..."
aws ec2 wait instance-running --region $Region --instance-ids $instanceId | Out-Null
$publicIp = aws ec2 describe-instances --region $Region --instance-ids $instanceId `
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text

Write-Host ""
Write-Host "接続:" -ForegroundColor Cyan
Write-Host "  ssh -i <$KeyName の秘密鍵> ubuntu@$publicIp"
Write-Host ""
Write-Host "インスタンス内で:" -ForegroundColor Cyan
Write-Host @"
  git clone <contamlab のパス or リモート> contamlab && cd contamlab
  bash scripts/10-bootstrap.sh
  bash scripts/20-rebuild-benchmark.sh
  bash scripts/30-record-environment.sh
"@
Write-Host ""
Write-Host "★ 止めるとき(課金を止める):" -ForegroundColor Yellow
Write-Host "  aws ec2 terminate-instances --region $Region --instance-ids $instanceId"
Write-Host "  ★ terminate する前に reports/ を手元へ回収すること(scp)。"
Write-Host "    キャッシュ data/cache/*.jsonl も回収する。作り直すと GPU 時間を捨てることになる。"
