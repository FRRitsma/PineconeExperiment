from pathlib import Path

from settings import DownloadSettings
from settings import INDEX_NAME
from src.database_functions import create_index_no_overwrite
from src.database_functions import create_list_of_upload_chunks
from src.database_functions import extract_image_paths_from_database_query
from src.database_functions import index_exists
from src.database_functions import return_random_vector_and_its_nearest_neighbors
from src.database_functions import upsert_batches
from src.download_dataset import download_imagenette2_dataset
from src.embedding import Embedder
from src.extract import extract_images_with_metadata
from src.extract import LabelPath
from src.visualization import visualize_similarities

if __name__ == "__main__":
    # Low level settings:
    N_LABELS = 10
    N_IMAGES = 1000
    CHUNK_SIZE = 50

    # Download data set if it does not exist:
    data_folder: Path = Path(__file__).parent / "data"
    imagenette2_folder: Path = data_folder / DownloadSettings.small_size.name_no_extension
    if not imagenette2_folder.is_dir():
        download_imagenette2_dataset(DownloadSettings.small_size, data_folder)

    # Create and fill the database if it does not exist:
    if not index_exists():
        DIMENSION = Embedder().embedding_dimension
        create_index_no_overwrite(DIMENSION)
        # Embed all images in train:
        embedder = Embedder()
        train_data = extract_images_with_metadata(N_LABELS, N_IMAGES, LabelPath.train)
        list_of_upload_chunks = create_list_of_upload_chunks(CHUNK_SIZE, train_data)
        # Upsert vectors:
        upsert_batches(list_of_upload_chunks, INDEX_NAME)

    # Get random vectors:
    random_vector, nearest_vectors = return_random_vector_and_its_nearest_neighbors(9)
    # Visualize the image and its most similar images:
    visualize_similarities(3, extract_image_paths_from_database_query(random_vector, nearest_vectors))
