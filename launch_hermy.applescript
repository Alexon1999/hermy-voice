on run
    set launchCommand to "/bin/launchctl kickstart gui/$(/usr/bin/id -u)/dev.ualexon.hermy-voice"
    try
        do shell script launchCommand
    on error errorText
        display dialog "Impossible de lancer Hermy : " & errorText buttons {"OK"} default button "OK"
    end try
end run
