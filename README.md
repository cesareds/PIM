# PIM
Trabalho de PIM - Implementação da segmentação por área

# Fluxo do programa
  Primeiramente, a fim de tornar a imagem binária é encontrado o threshold com a funcao find_threshold(image), e em seguida é aplicado o limiar usando a funcao da biblioteca numpy np.where(np.array(image)>threshold, 255, 0). Dessa maneira, é obtido uma matriz que consiste de 255 (branco) ou 0 (preto). Esta facilita a procura de grupos.
  Então é encontrado o centro de massa da imagem, e em seguida aplicado o algoritmo BFS para busca em profundidade na matriz, encontrando grupos e encontrando o grupo que mais se aproxima do centro de massa da imagem. Por fim é gerada a imagem com apenas o grupo encontrado destacado, deixando em preto todos os outros pixels, usando a função get_image_from_group(group).
  Assim, é mostrada a imagem na tela usando a biblioteca matplotlib.pyplot
