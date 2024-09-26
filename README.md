# Image Watermarking and Denoising Application

This Flask-based web application provides functionality for image denoising and digital watermarking. It allows users to upload images, apply various denoising techniques, embed encrypted watermarks, and extract watermarks from processed images.

## Features

- Image upload and processing
- Multiple denoising techniques:
  - Median filter
  - Gaussian filter
  - Laplacian filter
- Automatic noise type detection
- Digital watermarking using two methods:
  - Least Significant Bit (LSB)
  - Discrete Wavelet Transform (DWT)
- Encryption of watermark information
- Extraction and decryption of watermarks
- Performance metrics calculation (SSIM, PSNR, BER)
- User input validation for watermark information

## Requirements

- Python 3.7+
- Flask
- OpenCV (cv2)
- NumPy
- scikit-image
- PyWavelets
- Pillow
- cryptography

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/image-watermarking-app.git
   cd image-watermarking-app
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`.

3. Use the web interface to upload an image, enter watermark information, and process the image.

4. View the results of denoising and watermarking on the results page.

5. Use the extraction page to extract and decrypt watermarks from processed images.

## Project Structure

- `app.py`: Main Flask application file
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, JavaScript, uploaded images)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)
- [scikit-image](https://scikit-image.org/)
- [PyWavelets](https://pywavelets.readthedocs.io/)

## Disclaimer

This application is for educational and research purposes only. Ensure you have the right to modify and embed information in images before using this tool.