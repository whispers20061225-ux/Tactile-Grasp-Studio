param(
    [string]$RosSetup = "C:\\pixi_ws\\ros2-windows\\ros2-windows\\local_setup.bat",
    [string]$WorkspaceSetup = "",
    [int]$DomainId = 0,
    [string]$RealsenseSerial = "",
    [int]$ColorWidth = 640,
    [int]$ColorHeight = 480,
    [int]$ColorFps = 30,
    [int]$DepthWidth = 640,
    [int]$DepthHeight = 480,
    [int]$DepthFps = 30,
    [bool]$AlignDepth = $true,
    [bool]$WarmupRosGraph = $false,
    [int]$TopicTimeoutSec = 20,
    [int]$HzSampleSec = 10,
    [double]$MinColorHz = 3.0,
    [double]$MinDepthHz = 3.0,
    [int]$StartupRetryCount = 3,
    [int]$RetryBackoffSec = 4,
    [int]$LauncherTimeoutSec = 25
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Step([string]$msg) { Write-Host "[STEP] $msg" }
function Write-Ok([string]$msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-WarnMsg([string]$msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Fail([string]$msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red }

function Get-PreferredShell {
    if (Get-Command pwsh -ErrorAction SilentlyContinue) {
        return "pwsh"
    }
    return "powershell"
}

function New-Ros2CommandExpression {
    param([string[]]$Args)

    $quotedArgs = @(
        $Args | ForEach-Object { "'" + ([string]$_ -replace "'", "''") + "'" }
    ) -join ", "
    return "& { & ros2 @($quotedArgs) }"
}

function Start-Ros2CommandProcess {
    param(
        [string[]]$Args,
        [string]$StdoutPath,
        [string]$StderrPath
    )

    $shellExe = Get-PreferredShell
    $command = New-Ros2CommandExpression -Args $Args
    return Start-Process -FilePath $shellExe `
        -ArgumentList @("-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $command) `
        -PassThru `
        -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath `
        -WindowStyle Hidden
}

function Invoke-Ros2CommandCapture {
    param(
        [string[]]$Args,
        [int]$TimeoutSec
    )

    $outFile = Join-Path $env:TEMP ("programme_ros2_" + [guid]::NewGuid().ToString() + ".out.log")
    $errFile = Join-Path $env:TEMP ("programme_ros2_" + [guid]::NewGuid().ToString() + ".err.log")
    try {
        $proc = Start-Ros2CommandProcess -Args $Args -StdoutPath $outFile -StderrPath $errFile
        $exited = $proc.WaitForExit($TimeoutSec * 1000)
        if (-not $exited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }

        return [PSCustomObject]@{
            Exited   = $exited
            ExitCode = if ($exited) { $proc.ExitCode } else { $null }
            Stdout   = Read-TextSafe -Path $outFile
            Stderr   = Read-TextSafe -Path $errFile
        }
    }
    catch {
        return [PSCustomObject]@{
            Exited   = $false
            ExitCode = $null
            Stdout   = ""
            Stderr   = ""
        }
    }
    finally {
        Remove-Item $outFile, $errFile -Force -ErrorAction SilentlyContinue
    }
}

function Read-TextSafe {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return ""
    }
    $contentObj = Get-Content -Raw -Path $Path -ErrorAction SilentlyContinue
    if ($null -eq $contentObj) {
        return ""
    }
    return ([string]$contentObj).Trim()
}

function Get-PreferredPythonCommand {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return [PSCustomObject]@{ FilePath = "python"; PrefixArgs = @() }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return [PSCustomObject]@{ FilePath = "py"; PrefixArgs = @("-3") }
    }
    return $null
}

function Invoke-RealsenseStreamProbe {
    param(
        [string]$ColorTopic,
        [string]$DepthTopic,
        [string]$InfoTopic,
        [int]$FirstTimeoutSec,
        [int]$SampleSec
    )

    $pythonCommand = Get-PreferredPythonCommand
    if ($null -eq $pythonCommand) {
        return [PSCustomObject]@{
            Invoked = $false
            Success = $false
            Message = "python executable unavailable for RealSense probe"
            Result  = $null
            Stdout  = ""
            Stderr  = ""
        }
    }

    $probeScript = Join-Path $scriptDir "realsense_stream_probe.py"
    if (-not (Test-Path $probeScript)) {
        return [PSCustomObject]@{
            Invoked = $false
            Success = $false
            Message = "probe script not found: $probeScript"
            Result  = $null
            Stdout  = ""
            Stderr  = ""
        }
    }

    $timeoutSec = [Math]::Max(10, $FirstTimeoutSec + $SampleSec + 8)
    $outFile = Join-Path $env:TEMP ("programme_rs_probe_" + [guid]::NewGuid().ToString() + ".out.log")
    $errFile = Join-Path $env:TEMP ("programme_rs_probe_" + [guid]::NewGuid().ToString() + ".err.log")
    $argList = @()
    $argList += $pythonCommand.PrefixArgs
    $argList += @(
        $probeScript,
        "--color-topic", $ColorTopic,
        "--depth-topic", $DepthTopic,
        "--info-topic", $InfoTopic,
        "--first-timeout-sec", ([string]$FirstTimeoutSec),
        "--sample-sec", ([string]$SampleSec)
    )

    try {
        $proc = Start-Process -FilePath $pythonCommand.FilePath -ArgumentList $argList -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile -WindowStyle Hidden
        $exited = $proc.WaitForExit($timeoutSec * 1000)
        if (-not $exited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            return [PSCustomObject]@{
                Invoked = $false
                Success = $false
                Message = "probe timed out after ${timeoutSec}s"
                Result  = $null
                Stdout  = Read-TextSafe -Path $outFile
                Stderr  = Read-TextSafe -Path $errFile
            }
        }

        $stdout = Read-TextSafe -Path $outFile
        $stderr = Read-TextSafe -Path $errFile
        if (-not $stdout) {
            return [PSCustomObject]@{
                Invoked = $false
                Success = $false
                Message = if ($stderr) { "probe produced no stdout" } else { "probe returned empty result" }
                Result  = $null
                Stdout  = $stdout
                Stderr  = $stderr
            }
        }

        try {
            $result = $stdout | ConvertFrom-Json
        }
        catch {
            return [PSCustomObject]@{
                Invoked = $false
                Success = $false
                Message = "probe JSON parse failed"
                Result  = $null
                Stdout  = $stdout
                Stderr  = $stderr
            }
        }

        return [PSCustomObject]@{
            Invoked = $true
            Success = [bool]$result.success
            Message = if ($result.reason) { [string]$result.reason } else { if ($result.success) { "ok" } else { "probe failed" } }
            Result  = $result
            Stdout  = $stdout
            Stderr  = $stderr
        }
    }
    finally {
        Remove-Item $outFile, $errFile -Force -ErrorAction SilentlyContinue
    }
}

function Format-RealsenseProbeMessage {
    param(
        [object]$ProbeResult,
        [string]$FallbackMessage = "probe failed"
    )

    if ($null -eq $ProbeResult) {
        return $FallbackMessage
    }

    $missing = @()
    foreach ($name in @("color", "depth", "camera_info")) {
        $stream = $ProbeResult.streams.$name
        if ($null -eq $stream -or -not [bool]$stream.received) {
            $missing += $name
        }
    }

    $publisherSummary = @()
    foreach ($name in @("color", "depth", "camera_info")) {
        $publishers = @($ProbeResult.diagnostics.publishers.$name)
        $publisherSummary += ("{0}:{1}" -f $name, $publishers.Count)
    }

    $reason = if ($ProbeResult.reason) { [string]$ProbeResult.reason } else { $FallbackMessage }
    if ($missing.Count -gt 0) {
        return ("{0}; missing={1}; publishers={2}" -f $reason, ($missing -join ","), ($publisherSummary -join " "))
    }
    return ("{0}; publishers={1}" -f $reason, ($publisherSummary -join " "))
}
function Get-TopicListSafe {
    param(
        [int]$CommandTimeoutSec = 4
    )

    $timeoutSec = [Math]::Max(1, $CommandTimeoutSec)
    $result = Invoke-Ros2CommandCapture -Args @("topic", "list") -TimeoutSec $timeoutSec
    if ((-not $result.Exited) -or ($result.ExitCode -ne 0) -or (-not $result.Stdout)) {
        return @()
    }

    return @(
        $result.Stdout -split "`r?`n" |
            ForEach-Object { ([string]$_).Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
}

function Test-TopicMatch {
    param(
        [string[]]$Topics,
        [string]$TopicName
    )

    $topicAlt = $TopicName.TrimStart('/')
    $topicWithSlash = if ($topicAlt) { "/$topicAlt" } else { $TopicName }
    return (($Topics -contains $TopicName) -or ($Topics -contains $topicAlt) -or ($Topics -contains $topicWithSlash))
}

function Wait-Topic {
    param(
        [string]$TopicName,
        [int]$TimeoutSec
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $remainingSec = [int][Math]::Ceiling(($deadline - (Get-Date)).TotalSeconds)
        if ($remainingSec -le 0) {
            break
        }
        $commandTimeoutSec = [Math]::Max(1, [Math]::Min(4, $remainingSec))
        $topics = Get-TopicListSafe -CommandTimeoutSec $commandTimeoutSec
        if (Test-TopicMatch -Topics $topics -TopicName $TopicName) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }

    $finalTopics = Get-TopicListSafe -CommandTimeoutSec ([Math]::Max(1, [Math]::Min(4, $TimeoutSec)))
    return (Test-TopicMatch -Topics $finalTopics -TopicName $TopicName)
}

function Wait-MessageOnce {
    param(
        [string]$TopicName,
        [int]$TimeoutSec
    )

    $outFile = Join-Path $env:TEMP ("programme_echo_" + [guid]::NewGuid().ToString() + ".out.log")
    $errFile = Join-Path $env:TEMP ("programme_echo_" + [guid]::NewGuid().ToString() + ".err.log")
    try {
        $proc = Start-Ros2CommandProcess -Args @("topic", "echo", $TopicName, "--once") -StdoutPath $outFile -StderrPath $errFile
        $exited = $proc.WaitForExit($TimeoutSec * 1000)
        if (-not $exited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            return $false
        }
        return ($proc.ExitCode -eq 0)
    }
    finally {
        Remove-Item $outFile, $errFile -Force -ErrorAction SilentlyContinue
    }
}

function Get-TopicAverageHz {
    param(
        [string]$TopicName,
        [int]$SampleSec
    )

    $outFile = Join-Path $env:TEMP ("programme_hz_" + [guid]::NewGuid().ToString() + ".out.log")
    $errFile = Join-Path $env:TEMP ("programme_hz_" + [guid]::NewGuid().ToString() + ".err.log")
    try {
        $proc = Start-Ros2CommandProcess -Args @("topic", "hz", $TopicName) -StdoutPath $outFile -StderrPath $errFile
        Start-Sleep -Seconds $SampleSec
        if (-not $proc.HasExited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
        $proc.WaitForExit(2000) | Out-Null

        $raw = ""
        if (Test-Path $outFile) { $raw += (Get-Content -Raw $outFile) + "`n" }
        if (Test-Path $errFile) { $raw += (Get-Content -Raw $errFile) + "`n" }

        $matches = [regex]::Matches($raw, "average rate:\s*([0-9]+(?:\.[0-9]+)?)")
        if ($matches.Count -eq 0) {
            return $null
        }
        $last = $matches[$matches.Count - 1].Groups[1].Value
        return [double]$last
    }
    finally {
        Remove-Item $outFile, $errFile -Force -ErrorAction SilentlyContinue
    }
}

function Stop-RealsenseProcesses {
    $patterns = @(
        "realsense_watchdog.ps1",
        "realsense_camera_node",
        "realsense2_camera_node",
        "realsense2_camera",
        "realsense_stream_probe.py",
        '"topic" "hz" "/camera/camera',
        '"topic" "echo" "/camera/camera',
        '"topic" "list"'
    )
    $killed = 0
    $procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
    foreach ($proc in $procs) {
        $cmd = [string]$proc.CommandLine
        if ([string]::IsNullOrWhiteSpace($cmd)) {
            continue
        }
        foreach ($pattern in $patterns) {
            if ($cmd -match [regex]::Escape($pattern)) {
                try {
                    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
                    $killed += 1
                }
                catch {
                    # ignore
                }
                break
            }
        }
    }
    if ($killed -gt 0) {
        Write-WarnMsg "stopped $killed existing RealSense process(es) before restart"
        Start-Sleep -Seconds 1
    }
}

function Invoke-StartRealsenseOnly {
    param(
        [string]$RosSetupPath,
        [string]$WorkspaceSetupPath,
        [int]$RosDomainId,
        [string]$SerialNo,
        [bool]$WarmupGraph,
        [int]$ColorWidth,
        [int]$ColorHeight,
        [int]$ColorFps,
        [int]$DepthWidth,
        [int]$DepthHeight,
        [int]$DepthFps,
        [bool]$AlignDepth,
        [bool]$UseWatchdog = $false,
        [int]$TimeoutSec
    )

    $shellExe = Get-PreferredShell
    $scriptPath = Join-Path $scriptDir "start_realsense_only.ps1"
    $outFile = Join-Path $env:TEMP ("programme_rs_launch_" + [guid]::NewGuid().ToString() + ".out.log")
    $errFile = Join-Path $env:TEMP ("programme_rs_launch_" + [guid]::NewGuid().ToString() + ".err.log")
    $quotedScriptPath = "'" + ($scriptPath -replace "'", "''") + "'"
    $quotedRosSetupPath = "'" + ($RosSetupPath -replace "'", "''") + "'"
    $warmupToken = if ($WarmupGraph) { '$true' } else { '$false' }
    $watchdogToken = if ($UseWatchdog) { '$true' } else { '$false' }
    $alignDepthToken = if ($AlignDepth) { '$true' } else { '$false' }
    $launchCommand = "& { & $quotedScriptPath -RosSetup $quotedRosSetupPath -DomainId $RosDomainId -WarmupRosGraph $warmupToken -UseRealsenseWatchdog:$watchdogToken -Execute"
    if ($WorkspaceSetupPath) {
        $quotedWorkspaceSetupPath = "'" + ($WorkspaceSetupPath -replace "'", "''") + "'"
        $launchCommand += " -WorkspaceSetup $quotedWorkspaceSetupPath"
    }
    if ($SerialNo) {
        $quotedSerialNo = "'" + ($SerialNo -replace "'", "''") + "'"
        $launchCommand += " -RealsenseSerial $quotedSerialNo"
    }
    $launchCommand += " -ColorWidth $ColorWidth -ColorHeight $ColorHeight -ColorFps $ColorFps"
    $launchCommand += " -DepthWidth $DepthWidth -DepthHeight $DepthHeight -DepthFps $DepthFps"
    $launchCommand += " -AlignDepth $alignDepthToken"
    $launchCommand += " }"
    $argList = @(
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        $launchCommand
    )

    try {
        $proc = Start-Process -FilePath $shellExe -ArgumentList $argList -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile -WindowStyle Hidden
        $exited = $proc.WaitForExit($TimeoutSec * 1000)

        if (-not $exited) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            $stdout = Read-TextSafe -Path $outFile
            $stderr = Read-TextSafe -Path $errFile
            return [PSCustomObject]@{
                Success = $false
                Message = "launcher timed out after ${TimeoutSec}s"
                Stdout  = $stdout
                Stderr  = $stderr
            }
        }

        $stdout = Read-TextSafe -Path $outFile
        $stderr = Read-TextSafe -Path $errFile
        $exitCodeText = ""
        try {
            $exitCodeText = [string]$proc.ExitCode
        }
        catch {
            $exitCodeText = ""
        }
        $parsedExitCode = 0
        $hasExitCode = [int]::TryParse($exitCodeText, [ref]$parsedExitCode)

        if ($hasExitCode -and $parsedExitCode -ne 0) {
            return [PSCustomObject]@{
                Success = $false
                Message = "launcher exit code $parsedExitCode"
                Stdout  = $stdout
                Stderr  = $stderr
            }
        }
        if (-not $hasExitCode -and $stderr) {
            return [PSCustomObject]@{
                Success = $false
                Message = "launcher produced stderr (exit code unavailable)"
                Stdout  = $stdout
                Stderr  = $stderr
            }
        }
        return [PSCustomObject]@{
            Success = $true
            Message = if ($hasExitCode) { "launcher exited cleanly (code=$parsedExitCode)" } else { "launcher exited (code unavailable)" }
            Stdout  = $stdout
            Stderr  = $stderr
        }
    }
    finally {
        Remove-Item $outFile, $errFile -Force -ErrorAction SilentlyContinue
    }
}

function Test-RealsenseReady {
    param(
        [string]$ColorTopic,
        [string]$DepthTopic,
        [string]$InfoTopic,
        [int]$TopicWaitSec,
        [int]$SampleWindowSec,
        [double]$MinColor,
        [double]$MinDepth
    )

    Write-Step "probing RealSense streams via rclpy (first-message timeout=${TopicWaitSec}s, sample=${SampleWindowSec}s)"
    $probe = Invoke-RealsenseStreamProbe -ColorTopic $ColorTopic -DepthTopic $DepthTopic -InfoTopic $InfoTopic -FirstTimeoutSec $TopicWaitSec -SampleSec $SampleWindowSec

    if (-not $probe.Invoked) {
        return [PSCustomObject]@{
            Success = $false
            Message = $probe.Message
            ColorHz = $null
            DepthHz = $null
            Probe = $null
            ProbeStdout = $probe.Stdout
            ProbeStderr = $probe.Stderr
        }
    }
    if (-not $probe.Success) {
        return [PSCustomObject]@{
            Success = $false
            Message = (Format-RealsenseProbeMessage -ProbeResult $probe.Result -FallbackMessage $probe.Message)
            ColorHz = $null
            DepthHz = $null
            Probe = $probe.Result
            ProbeStdout = $probe.Stdout
            ProbeStderr = $probe.Stderr
        }
    }

    $colorHz = $probe.Result.streams.color.sample_hz
    $depthHz = $probe.Result.streams.depth.sample_hz
    if ($null -eq $colorHz) {
        Write-WarnMsg "probe succeeded but color hz is unavailable; keeping RealSense node running"
        return [PSCustomObject]@{
            Success = $true
            Message = "probe passed; color hz unavailable"
            ColorHz = $null
            DepthHz = $depthHz
            Probe = $probe.Result
            ProbeStdout = $probe.Stdout
            ProbeStderr = $probe.Stderr
        }
    }
    if ($null -eq $depthHz) {
        Write-WarnMsg "probe succeeded but depth hz is unavailable; keeping RealSense node running"
        return [PSCustomObject]@{
            Success = $true
            Message = "probe passed; depth hz unavailable"
            ColorHz = $colorHz
            DepthHz = $null
            Probe = $probe.Result
            ProbeStdout = $probe.Stdout
            ProbeStderr = $probe.Stderr
        }
    }
    if ([double]$colorHz -lt $MinColor) {
        return [PSCustomObject]@{
            Success = $false
            Message = ("color hz {0:N3} < min {1:N3}" -f $colorHz, $MinColor)
            ColorHz = [double]$colorHz
            DepthHz = [double]$depthHz
            Probe = $probe.Result
            ProbeStdout = $probe.Stdout
            ProbeStderr = $probe.Stderr
        }
    }
    if ([double]$depthHz -lt $MinDepth) {
        return [PSCustomObject]@{
            Success = $false
            Message = ("depth hz {0:N3} < min {1:N3}" -f $depthHz, $MinDepth)
            ColorHz = [double]$colorHz
            DepthHz = [double]$depthHz
            Probe = $probe.Result
            ProbeStdout = $probe.Stdout
            ProbeStderr = $probe.Stderr
        }
    }
    return [PSCustomObject]@{
        Success = $true
        Message = "ok"
        ColorHz = [double]$colorHz
        DepthHz = [double]$depthHz
        Probe = $probe.Result
        ProbeStdout = $probe.Stdout
        ProbeStderr = $probe.Stderr
    }
}

Write-Step "loading ROS2 Windows environment"
. (Join-Path $scriptDir "env_ros2_windows.ps1") -RosSetup $RosSetup -WorkspaceSetup $WorkspaceSetup -DomainId $DomainId -WarmupRosGraph $WarmupRosGraph

if (-not (Get-Command ros2 -ErrorAction SilentlyContinue)) {
    Write-Fail "ros2 command is unavailable after environment setup."
    exit 1
}

$colorTopic = "/camera/camera/color/image_raw"
$depthTopic = "/camera/camera/aligned_depth_to_color/image_raw"
$infoTopic = "/camera/camera/color/camera_info"

$lastResult = $null
for ($attempt = 1; $attempt -le $StartupRetryCount; $attempt++) {
    Write-Step "RealSense startup attempt $attempt/$StartupRetryCount"
    Stop-RealsenseProcesses

    Write-Step "spawning RealSense launcher (timeout=${LauncherTimeoutSec}s)"
    $launchResult = Invoke-StartRealsenseOnly `
        -RosSetupPath $RosSetup `
        -WorkspaceSetupPath $WorkspaceSetup `
        -RosDomainId $DomainId `
        -SerialNo $RealsenseSerial `
        -WarmupGraph $WarmupRosGraph `
        -ColorWidth $ColorWidth `
        -ColorHeight $ColorHeight `
        -ColorFps $ColorFps `
        -DepthWidth $DepthWidth `
        -DepthHeight $DepthHeight `
        -DepthFps $DepthFps `
        -AlignDepth $AlignDepth `
        -UseWatchdog $false `
        -TimeoutSec $LauncherTimeoutSec
    if (-not $launchResult.Success) {
        Write-WarnMsg ("attempt $attempt launcher failed: " + $launchResult.Message)
        if ($launchResult.Stdout) {
            Write-Host "[DIAG][launcher stdout]"
            Write-Host $launchResult.Stdout
        }
        if ($launchResult.Stderr) {
            Write-Host "[DIAG][launcher stderr]"
            Write-Host $launchResult.Stderr
        }
        $lastResult = [PSCustomObject]@{
            Success = $false
            Message = "launcher failed: $($launchResult.Message)"
            ColorHz = $null
            DepthHz = $null
        }
        if ($attempt -lt $StartupRetryCount) {
            Start-Sleep -Seconds $RetryBackoffSec
            continue
        }
        break
    }

    Write-Ok "RealSense launcher returned, waiting for graph stabilization"

    Start-Sleep -Seconds 3

    $lastResult = Test-RealsenseReady `
        -ColorTopic $colorTopic `
        -DepthTopic $depthTopic `
        -InfoTopic $infoTopic `
        -TopicWaitSec $TopicTimeoutSec `
        -SampleWindowSec $HzSampleSec `
        -MinColor $MinColorHz `
        -MinDepth $MinDepthHz

    if ($lastResult.Success) {
        if ($null -ne $lastResult.ColorHz -and $null -ne $lastResult.DepthHz) {
            Write-Ok ("RealSense READY: color={0:N3}Hz depth={1:N3}Hz" -f $lastResult.ColorHz, $lastResult.DepthHz)
        } else {
            Write-Ok ("RealSense READY: {0}" -f $lastResult.Message)
        }
        Write-Host "[READY] You can now start VM one-click debug script."
        exit 0
    }

    Write-WarnMsg ("attempt $attempt failed: " + $lastResult.Message)
    if ($attempt -lt $StartupRetryCount) {
        Start-Sleep -Seconds $RetryBackoffSec
    }
}

$lastMessage = if ($lastResult -and $lastResult.Message) { $lastResult.Message } else { "unknown failure" }
Write-Fail ("RealSense did not reach ready state after {0} attempts. Last error: {1}" -f $StartupRetryCount, $lastMessage)
if ($lastResult -and $lastResult.Probe) {
    Write-Host "[DIAG] probe diagnostics:"
    try {
        $lastResult.Probe | ConvertTo-Json -Depth 8 | Write-Host
    }
    catch {
        Write-WarnMsg "unable to print probe diagnostics"
    }
} elseif ($lastResult -and ($lastResult.ProbeStdout -or $lastResult.ProbeStderr)) {
    if ($lastResult.ProbeStdout) {
        Write-Host "[DIAG][probe stdout]"
        Write-Host $lastResult.ProbeStdout
    }
    if ($lastResult.ProbeStderr) {
        Write-Host "[DIAG][probe stderr]"
        Write-Host $lastResult.ProbeStderr
    }
}
exit 1
