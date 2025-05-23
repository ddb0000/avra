# Function to display the menu and get user choice
function Show-Menu {
    param(
        [string[]]$MenuItems
    )
    Write-Host "Please select an option:"
    for ($i = 0; $i -lt $MenuItems.Count; $i++) {
        Write-Host "[" $($i + 1) "] $($MenuItems[$i])"
    }
    Write-Host "[0] Exit"
    $choice = Read-Host "Enter your choice"
    return $choice
}

# Function to execute a script or file
function Execute-Item {
    param(
        [string]$ItemPath
    )
    Write-Host "Executing: $($ItemPath)"
    # Check if it's a Python file
    if ($ItemPath -like "*.py") {
        # Use python.exe to run Python scripts
        & python.exe "$ItemPath"
    }
    # You can add more conditions here for other file types if needed
    # For example, to run a PowerShell script:
    # elseif ($ItemPath -like "*.ps1") {
    #     & powershell.exe -File "$ItemPath"
    # }
    # Assuming other files in googleGenAI might be executables or scripts
    else {
        # Use the call operator '&' to execute the item
        & "$ItemPath"
    }

    # Check the exit code of the last executed command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error executing $($ItemPath). Exit code: $($LASTEXITCODE)" -ForegroundColor Red
    } else {
        Write-Host "Execution of $($ItemPath) completed successfully." -ForegroundColor Green
    }
    Write-Host "`nPress Enter to continue..."
    [void] $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") # Wait for a key press without displaying it
}

# Find python scripts in the current folder
# and any files in the googleGenAI subfolder
$menu_items = @()

# Find Python files in the current directory
$pythonFiles = Get-ChildItem -Path . -Filter "*.py" -File -ErrorAction SilentlyContinue
$menu_items += $pythonFiles.FullName

# Find files in the googleGenAI subfolder
# Use -ErrorAction SilentlyContinue in case the folder doesn't exist
$googleGenAIFiles = Get-ChildItem -Path .\googleGenAI\* -File -ErrorAction SilentlyContinue
$menu_items += $googleGenAIFiles.FullName

# Check if any items were found
if ($menu_items.Count -eq 0) {
    Write-Host "No Python scripts found in the current directory or files in the googleGenAI subfolder."
    exit 1
}

# Display menu and execute user choice
while ($true) {
    Clear-Host # Clear the console for a cleaner menu display
    $choice = Show-Menu -MenuItems $menu_items

    if ($choice -eq "0") {
        Write-Host "Exiting..."
        break # Exit the loop
    } elseif ($choice -ge 1 -and $choice -le $menu_items.Count) {
        # Adjust index for 0-based array
        $item_index = $choice - 1
        Execute-Item -ItemPath $menu_items[$item_index]
    } else {
        Write-Host "Invalid choice. Please try again." -ForegroundColor Yellow
        Write-Host "`nPress Enter to continue..."
        [void] $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") # Wait for a key press
    }
}
