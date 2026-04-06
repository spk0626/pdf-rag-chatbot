$ErrorActionPreference = 'Stop'

$backendPort = 8000
$frontendPort = 5173
$backendUrl = "http://127.0.0.1:$backendPort"
$frontendUrl = "http://127.0.0.1:$frontendPort"

$backendProcess = $null
$frontendProcess = $null

function Wait-ForHttp {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [int]$MaxAttempts = 40,
        [int]$DelayMs = 500
    )

    for ($i = 1; $i -le $MaxAttempts; $i++) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return $true
            }
        } catch {
            # Retry until max attempts.
        }
        Start-Sleep -Milliseconds $DelayMs
    }

    return $false
}

try {
    Write-Host "[smoke] Starting backend..."
    $backendProcess = Start-Process -FilePath "python" -ArgumentList "-m uvicorn api.main:app --host 127.0.0.1 --port $backendPort" -PassThru -WindowStyle Hidden

    if (-not (Wait-ForHttp -Url "$backendUrl/health")) {
        throw "Backend did not become ready at $backendUrl/health"
    }

    Write-Host "[smoke] Starting frontend..."
    $frontendProcess = Start-Process -FilePath "npm" -ArgumentList "run dev -- --host 127.0.0.1 --port $frontendPort" -PassThru -WindowStyle Hidden

    if (-not (Wait-ForHttp -Url $frontendUrl)) {
        throw "Frontend did not become ready at $frontendUrl"
    }

    $health = Invoke-WebRequest -Uri "$backendUrl/health" -UseBasicParsing -TimeoutSec 5
    if ($health.StatusCode -ne 200) {
        throw "Health endpoint returned $($health.StatusCode)"
    }

    Write-Host "[smoke] Backend and frontend are reachable."
    Write-Host "[smoke] PASS"
    exit 0
}
catch {
    Write-Error "[smoke] FAIL: $($_.Exception.Message)"
    exit 1
}
finally {
    if ($frontendProcess -and -not $frontendProcess.HasExited) {
        Stop-Process -Id $frontendProcess.Id -Force
    }
    if ($backendProcess -and -not $backendProcess.HasExited) {
        Stop-Process -Id $backendProcess.Id -Force
    }
}
