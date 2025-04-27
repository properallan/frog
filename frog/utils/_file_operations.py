def create_clean_directory(dir_path, overwrite=True):
    """
    Cria um diretório. Se já existir, remove todo o conteúdo e recria do zero.

    Args:
        dir_path (str): Caminho do diretório a ser criado/limpo.
    """
    import os
    import shutil
    
    if os.path.exists(dir_path) and overwrite:
        shutil.rmtree(dir_path)
        #print(f'REMOVENDO O DIRETORIO: {dir_path}')  # Remove o diretório e todo o conteúdo
    os.makedirs(dir_path, exist_ok=True)  # Recria o diretório vazio
