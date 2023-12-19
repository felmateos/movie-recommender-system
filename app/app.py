import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

### Lógica do sistema

ratings = pd.read_csv('data/reduced/ratings_m10.csv')
ratings.reindex()
movies = pd.read_csv('data/reduced/movies_m10_rich_pre.csv', index_col='movieId')
movies_title = movies[['title']]

ratings_movies = ratings.merge(movies_title, on='movieId')

def train_test_column_split(df, group_column, split_column, y_label, train_size):
    df = df.sort_values(by=split_column, ascending=True)   
    train = pd.DataFrame(columns=df.columns)
    test = pd.DataFrame(columns=df.columns)

    for idx in df[group_column].unique():
        group = df.loc[df[group_column] == idx]

        q_user = group[group[split_column].le(group[split_column].quantile(train_size))]
        p_user = group[group[split_column].ge(group[split_column].quantile(train_size))]

        train = pd.concat([train, q_user])
        test = pd.concat([test, p_user])
    train = train.sort_index(ascending=True)
    test = test.sort_index(ascending=True)

    X_labels = [c for c in df.columns]

    X_train = train[X_labels]
    X_test = test[X_labels]

    return (X_train, X_test)

X_train, X_test = train_test_column_split(ratings_movies, 'userId', 'timestamp', 'rating', .9)

user_movie_train = X_train.pivot(index='movieId', columns='userId', values='rating').fillna(0)
user_movie_test = X_test.pivot(index='movieId', columns='userId', values='rating').fillna(0)

def find_correlation_between_two_users(ratings_df: pd.DataFrame, user1: str, user2: str):
    """Find correlation between two users based on their rated movies using Pearson correlation"""
    rated_movies_by_both = ratings_df[[user1, user2]].dropna(axis=0).values
    user1_ratings = rated_movies_by_both[:, 0].reshape(1, -1)
    user2_ratings = rated_movies_by_both[:, 1].reshape(1, -1)
    return cosine_similarity(user1_ratings, user2_ratings)

def get_rated_user_for_a_movie(ratings_df: pd.DataFrame, movie: str):
    return ratings_df.loc[movie, :].dropna().index.values


def get_top_neighbors(
    similarity_df: pd.DataFrame, user: str, rated_users: str, n_neighbors: int
):
    return similarity_df[user][rated_users].nlargest(n_neighbors).to_dict()


def subtract_bias(rating: float, mean_rating: float):
    return rating - mean_rating


def get_neighbor_rating_without_bias_per_movie(
    ratings_df: pd.DataFrame, user: str, movie: str
):
    """Substract the rating of a user from the mean rating of that user to eliminate bias"""
    mean_rating = ratings_df[user].mean()
    rating = ratings_df.loc[movie, user]
    return subtract_bias(rating, mean_rating)


def get_ratings_of_neighbors(ratings_df: pd.DataFrame, neighbors: list, movie: str):
    """Get the ratings of all neighbors after adjusting for biases"""
    return [
        get_neighbor_rating_without_bias_per_movie(ratings_df, neighbor, movie)
        for neighbor in neighbors
    ]

def get_weighted_average_rating_of_neighbors(ratings: list, neighbor_distance: list):
    weighted_sum = np.array(ratings).dot(np.array(neighbor_distance))
    abs_neigbor_distance = np.abs(neighbor_distance)
    return weighted_sum / np.sum(abs_neigbor_distance)


def ger_user_rating(ratings_df: pd.DataFrame, user: str, avg_neighbor_rating: float):
    user_avg_rating = ratings_df[user].mean()
    return round(user_avg_rating + avg_neighbor_rating, 2)

def predict_rating(
    df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    user: str,
    movie: str,
    n_neighbors: int = 2,
):
    """Predict the rating of a user for a movie based on the ratings of neighbors"""
    ratings_df = df.copy()

    rated_users = get_rated_user_for_a_movie(ratings_df, movie)

    top_neighbors_distance = get_top_neighbors(
        similarity_df, user, rated_users, n_neighbors
    )
    neighbors, distance = top_neighbors_distance.keys(), top_neighbors_distance.values()

    #print(f"Top {n_neighbors} neighbors of user {user}, {movie}: {list(neighbors)}, distance: {list(distance)}")

    ratings = get_ratings_of_neighbors(ratings_df, neighbors, movie)
    avg_neighbor_rating = get_weighted_average_rating_of_neighbors(
        ratings, list(distance)
    )

    return ger_user_rating(ratings_df, user, avg_neighbor_rating)

def adjust_rating(nota):
    if nota < 0:
        return 0
    elif nota > 5:
        return 5
    else:
        # Arredonda para o valor mais próximo em incrementos de 0.5
        return round(nota * 2) / 2

def get_n_recommendations(user: int, n: int, user_movie_mat: pd.DataFrame, users_similarity_mat, movies: pd.DataFrame, n_neighbors: int):
    df = user_movie_mat.copy()
    recommendations = pd.DataFrame(columns=['movieId', 'title', 'pred_rating'])

    for movie, _ in df[user].items():
        if df.loc[movie, user] == 0:
            df.loc[movie, user] = predict_rating(user_movie_mat, users_similarity_mat, user, movie, n_neighbors)
            new_row = {'movieId': movie, 'title': movies.loc[movie]['title'], 'pred_rating': adjust_rating(df.loc[movie, user])}
            recommendations.loc[len(recommendations)] = new_row

    recommendations = recommendations.sort_values(by='pred_rating', ascending=False)
    return recommendations.head(n) if n > 0  else recommendations

# Função para gerar recomendações (substitua isso pelo seu script real)
def gerar_recomendacoes(usuario_selecionado):

    
    


    users_list = list(user_movie_train.columns)
    movies_list = list(user_movie_train.index)

    users_similarity_mat = pd.read_pickle('data/preprocessed/users_similarity_mat_cosim.pkl')

    recomendacoes_df = get_n_recommendations(usuario_selecionado, 10, user_movie_train, users_similarity_mat, movies, 30)
    return recomendacoes_df

# Configuração da página
st.set_page_config(
    page_title="EACHFLIX",
    page_icon="assets\icon.png",  # Substitua pelo caminho do ícone desejado
    layout="wide"
)

# Título e descrição do projeto
st.title("Projeto da Disciplina \"Análise de Redes Sociais\"")
st.image("assets\logo.png", use_column_width=True)  # Substitua pelo caminho da sua imagem de logo
st.markdown("""
# Bem-vindo à EACHFLIX!

Esta plataforma inovadora utiliza um sistema de recomendação baseado em filtro colaborativo com similaridade por cosseno. Ao integrar dados de filmes da Netflix obtidos do The Movie Database (TMDb), proporcionamos uma experiência personalizada, ajudando você a descobrir novos filmes com base nas suas preferências únicas.

## Como Funciona:

1. **Selecione seu Usuário:** Escolha um usuário entre os usuários disponíveis em nossa base de dados. Cada usuário representa diferentes gostos e preferências cinematográficas.

2. **Explore Recomendações Personalizadas:** Ao clicar no botão "Gerar Recomendações", nosso algoritmo de filtro colaborativo entra em ação, analisando padrões de visualização semelhantes entre usuários e recomendando filmes que o usuário selecionado provavelemente melhor avaliaria.

3. **Descubra Novos Filmes:** As recomendações são apresentadas em uma lista organizada, contendo informações sobre cada filme, como nome e a nota prevista. Explore essas sugestões personalizadas e encontre filmes que correspondam aos seus interesses.

## Sobre o Projeto:

Este projeto é um esforço colaborativo para oferecer uma experiência única de descoberta de filmes. Utilizando técnicas avançadas de similaridade por cosseno.

""")

# Seleção de usuário
usuarios_base_de_dados = user_movie_train.columns  # Substitua pelos seus usuários reais
usuario_selecionado = st.selectbox("Selecione um usuário", usuarios_base_de_dados)

# Exibição de recomendações
if st.button("Gerar Recomendações"):
    recomendacoes_df = gerar_recomendacoes(usuario_selecionado)
    st.subheader("Recomendações para o usuário {}:".format(usuario_selecionado))
    st.dataframe(recomendacoes_df)

# Créditos
st.sidebar.markdown("### Créditos")
st.sidebar.text("""Autores: 
    Bruno F. Raquel
    Ezequiel Park
    Felipe M. Castro
    Rodrigo D. Ferreira""")
st.sidebar.text("""Docente: 
    Luciano Digiampietri""")

# Rodapé
st.sidebar.markdown("---")
st.sidebar.text("""Este projeto foi desenvolvido 
para a Disciplina \"Análise de Redes Sociais\"""")

# Execute o aplicativo usando o seguinte comando no terminal:
# streamlit run nome_do_arquivo.py



