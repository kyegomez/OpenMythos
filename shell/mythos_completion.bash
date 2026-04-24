#!/bin/bash
# OpenMythos Shell Completion - Bash
# Install: source this file or add to ~/.bashrc

_mythos_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    opts="--help --version --web --example --debug"
    examples="memory tools context evolution"
    
    case "${prev}" in
        --example|-e)
            COMPREPLY=($(compgen -W "${examples}" -- ${cur}))
            return 0
            ;;
        python|mythos.py)
            COMPREPLY=($(compgen -W "${opts}" -- ${cur}))
            return 0
            ;;
        *)
            ;;
    esac
    
    COMPREPLY=($(compgen -W "${opts}" -- ${cur}))
    return 0
}

complete -F _mythos_completion mythos.py
complete -F _mythos_completion mythos
