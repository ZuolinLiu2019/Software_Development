from neural_network_model import __version__ as _version
from neural_network_model.predict import make_single_prediction

def test_make_prediction_one_sample(charlock_dir):
    # Given
    filename = '1.png'
    expected_classification = 'Charlock'

    # When
    results = make_single_prediction(image_directory=charlock_dir,
                                    image_name=filename)

    # Then
    assert results['predictions'] is not None
    assert result['readable_predictions[0]'][0] == expected_classification
    assert result['version'] == _version
    
