# OpenMythos Shell Completion - Fish
# Install: copy to ~/.config/fish/completions/mythos.fish

complete -c mythos -s h -l help -d "Show help"
complete -c mythos -s v -l version -d "Show version"
complete -c mythos -s w -l web -d "Start web dashboard"
complete -c mythos -s e -l example -d "Run example" -a "memory tools context evolution"
complete -c mythos -l debug -d "Enable debug mode"
