# app.py
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import hashlib
import pywt
from PIL import Image
import io
import base64
from cryptography.fernet import Fernet, InvalidToken
import json
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def median_denoise(image):
    return cv2.medianBlur(image, 3)

def gaussian_denoise(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def laplacian_denoise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    abs_laplacian = cv2.convertScaleAbs(laplacian)
    denoised_image = cv2.addWeighted(image, 1, cv2.cvtColor(abs_laplacian, cv2.COLOR_GRAY2BGR), 0.5, 0)
    return denoised_image

def calculate_ssim(original, denoised):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    denoised_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(original_gray, denoised_gray, full=True)
    return score

def calculate_psnr(original, watermarked):
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ber(original_watermark, extracted_watermark):
    # Ensure both watermarks are the same length by padding the shorter one
    max_len = max(len(original_watermark), len(extracted_watermark))
    original_binary = ''.join(format(ord(char), '08b') for char in original_watermark.ljust(max_len))
    extracted_binary = ''.join(format(ord(char), '08b') for char in extracted_watermark.ljust(max_len))

    # Compare bit by bit
    errors = sum(o != e for o, e in zip(original_binary, extracted_binary))
    total_bits = len(original_binary)

    ber = errors / total_bits if total_bits > 0 else float('inf')  # Avoid division by zero
    return ber


# Encryption Functions
def generate_fernet_key(passkey):
    """
    Generate a Fernet key from the user-provided passkey using SHA-256.
    This function does not assume the passkey is hexadecimal.
    """
    # Hash the passkey with SHA-256 to generate a 32-byte key
    key = hashlib.sha256(passkey.encode('utf-8')).digest()
    # Fernet requires a base64-encoded 32-byte key, so we encode the hash result
    return base64.urlsafe_b64encode(key)

def encrypt_watermark(watermark_text):
    """
    Encrypt the watermark text using Fernet encryption with a generated key.
    """
    fernet_key = generate_fernet_key(watermark_text)  # Generate the Fernet key
    cipher_suite = Fernet(fernet_key)  # Initialize the cipher suite
    encrypted = cipher_suite.encrypt(watermark_text.encode('utf-8'))  # Encrypt the watermark
    return encrypted.decode('utf-8') 

def decrypt_watermark(encrypted_watermark, watermark_text):
    """
    Decrypt the encrypted watermark using Fernet encryption with the original watermark text as the key.
    """
    try:
        fernet_key = generate_fernet_key(watermark_text)  # Generate the same key used for encryption
        cipher_suite = Fernet(fernet_key)  # Initialize the cipher suite
        decrypted = cipher_suite.decrypt(encrypted_watermark.encode('utf-8')).decode('utf-8')  # Decrypt and decode
        return decrypted
    except InvalidToken:
        raise InvalidToken("Decryption failed. The hash key may be incorrect or the watermark may be corrupted.")
    except Exception as e:
        raise Exception(f"An error occurred during decryption: {str(e)}")
    
# Watermark Embedding and Extraction Functions
def lsb_watermark(image, encrypted_watermark):
    img = image.copy()
    height, width, _ = img.shape
    binary_watermark = ''.join([format(ord(char), '08b') for char in encrypted_watermark])
    idx = 0
    for i in range(height):
        for j in range(width):
            for k in range(3):  # B, G, R
                if idx < len(binary_watermark):
                    img[i,j,k] = (img[i,j,k] & ~1) | int(binary_watermark[idx])
                    idx += 1
                else:
                    break
    return img

def lsb_extract(image, length):
    binary_watermark = ""
    height, width, _ = image.shape
    total_bits = length * 8
    idx = 0
    for i in range(height):
        for j in range(width):
            for k in range(3):  # B, G, R
                if idx < total_bits:
                    binary_watermark += str(image[i,j,k] & 1)
                    idx += 1
                else:
                    break
    chars = [binary_watermark[i:i+8] for i in range(0, len(binary_watermark), 8)]
    try:
        watermark = ''.join([chr(int(char, 2)) for char in chars])
    except:
        watermark = ""
    return watermark

def wavelet_watermark(image, encrypted_watermark):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs

    LL_normalized = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX)
    LL_normalized = LL_normalized.astype(np.uint8)

    binary_watermark = ''.join([format(ord(char), '08b') for char in encrypted_watermark])
    wm_length = len(binary_watermark)

    if wm_length > LL_normalized.size:
        raise ValueError("Watermark is too long to embed in the image.")

    for i in range(wm_length):
        LL_normalized.flat[i] = (LL_normalized.flat[i] & ~1) | int(binary_watermark[i])

    LL_watermarked = LL_normalized.astype(np.float64)
    LL_watermarked = LL_watermarked / 255 * (LL.max() - LL.min()) + LL.min()

    coeffs_watermarked = (LL_watermarked, (LH, HL, HH))
    watermarked = pywt.idwt2(coeffs_watermarked, 'haar')
    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
    watermarked_bgr = cv2.cvtColor(watermarked, cv2.COLOR_RGB2BGR)
    return watermarked_bgr

def wavelet_extract(image, length):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs

    LL_normalized = cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX)
    LL_normalized = LL_normalized.astype(np.uint8)

    binary_watermark = ""
    total_bits = length * 8
    for i in range(total_bits):
        if i < LL_normalized.size:
            binary_watermark += str(LL_normalized.flat[i] & 1)
        else:
            break

    chars = [binary_watermark[i:i+8] for i in range(0, len(binary_watermark), 8)]
    try:
        watermark = ''.join([chr(int(char, 2)) for char in chars])
    except:
        watermark = ""
    return watermark

def validate_input(form_data):
    errors = []

    if any(char.isdigit() for char in form_data['full_name']):
        errors.append("Full Name cannot contain digits.")

    try:
        dob = datetime.strptime(form_data['dob'], '%Y-%m-%d')
        if dob > datetime.now():
            errors.append("Date of Birth cannot be in the future.")
    except ValueError:
        errors.append("Invalid Date of Birth format.")

    if form_data['gender'] not in ['M', 'F']:
        errors.append("Gender must be either 'M' or 'F'.")

    if not form_data['contact_number'].isdigit() or len(form_data['contact_number']) not in [10, 11]:
        errors.append("Contact Number must be 10 or 11 digits only.")

    if form_data['race'] not in ['Chinese', 'Malay', 'Indian']:
        errors.append("Race must be Chinese, Malay, or Indian.")

    return errors

# Hashing Functions and Encryption

def generate_fernet_key(hash_key):
    """
    Generate a Fernet key from the provided hash key.
    The hash key can be any string, and we will generate a SHA-256 digest from it.
    """
    # If the hash_key is a hex string, convert it to bytes first
    if isinstance(hash_key, str):
        hash_key_bytes = bytes.fromhex(hash_key)
    else:
        hash_key_bytes = hash_key
    # Use SHA-256 to hash the key to 32 bytes (required for Fernet)
    key = hashlib.sha256(hash_key_bytes).digest()
    return base64.urlsafe_b64encode(key)  # Return a base64-encoded key

import numpy as np
import cv2

def detect_noise_type(image):
    """Detect the type of noise present in the image"""
    # Check for salt-and-pepper noise by looking for extreme black and white pixels
    black_pixels = np.sum(image == 0)
    white_pixels = np.sum(image == 255)
    total_pixels = image.size
    
    # Heuristics to detect salt-and-pepper noise (thresholds can be adjusted)
    if (black_pixels + white_pixels) / total_pixels > 0.05:
        return 'salt_and_pepper'
    
    # Check for Gaussian noise by analyzing the pixel value distribution
    mean, stddev = cv2.meanStdDev(image)
    if stddev[0] > 20:  # This threshold can be tuned based on noise level
        return 'gaussian'

    # If no specific noise type detected, assume some high-frequency noise (Laplacian)
    return 'high_frequency'

def denoise_image_based_on_noise_type(image, denoise_results):
    """Apply the best denoising method based on detected noise type"""
    noise_type = detect_noise_type(image)
    
    if noise_type == 'salt_and_pepper':
        print("Detected salt-and-pepper noise, applying Median filter")
        best_method = 'Median'
    elif noise_type == 'gaussian':
        print("Detected Gaussian noise, applying Gaussian filter")
        best_method = 'Gaussian'
    else:
        print("Detected high-frequency noise, applying Laplacian filter")
        best_method = 'Laplacian'
    
    # Now get the best method based on SSIM score
    best_ssim = denoise_results[best_method]['ssim']
    threshold = 0.80  # Set threshold for SSIM

    if best_ssim < threshold:
        # If SSIM is below the threshold, revert to the original noisy image
        print(f"SSIM for {best_method} is less than {threshold * 100}%, reverting to original image.")
        processed_image = image  # Keep the original noisy image
        denoise_applied = False
    else:
        # Otherwise, use the best denoised image
        print(f"Using the best denoising method: {best_method} with SSIM: {best_ssim}")
        processed_image = cv2.imread(denoise_results[best_method]['image'])  # Apply the best filter
        denoise_applied = True
    
    return processed_image, denoise_applied


def encrypt_watermark(watermark_text, hash_key):
    """
    Encrypt the watermark text using Fernet encryption with the provided hash key.
    """
    fernet_key = generate_fernet_key(hash_key)  # Generate the Fernet key from the hash key
    cipher_suite = Fernet(fernet_key)  # Initialize the Fernet cipher
    encrypted = cipher_suite.encrypt(watermark_text.encode('utf-8'))  # Encrypt the watermark text
    return encrypted.decode('utf-8')  # Return the encrypted watermark as a string


def decrypt_watermark(encrypted_watermark, hash_key):
    """
    Decrypt the encrypted watermark using the provided hash key.
    """
    try:
        fernet_key = generate_fernet_key(hash_key)  # Generate the Fernet key from the hash key
        cipher_suite = Fernet(fernet_key)  # Initialize the Fernet cipher
        decrypted = cipher_suite.decrypt(encrypted_watermark.encode('utf-8')).decode('utf-8')  # Decrypt the watermark
        return decrypted  # Return the decrypted watermark as a string
    except InvalidToken:
        raise InvalidToken("Decryption failed. The hash key may be incorrect or the watermark may be corrupted.")
    except Exception as e:
        raise Exception(f"An error occurred during decryption: {str(e)}")


# Example of using the hash key (restoring original flow)
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            try:
                validation_errors = validate_input(request.form)
                if validation_errors:
                    return render_template('index.html', errors=validation_errors)

                # Retrieve the user-provided passkey
                passkey = request.form['passkey']
                if not passkey:
                    return render_template('index.html', error="Passkey is required for encryption.")

                # Prepare watermark information
                watermark_info = {
                    'full_name': request.form['full_name'],
                    'dob': request.form['dob'],
                    'gender': request.form['gender'],
                    'contact_number': request.form['contact_number'],
                    'race': request.form['race'],
                    'disease': request.form['disease']
                }
                watermark_text = json.dumps(watermark_info)

                # Encrypt the watermark with the user-provided passkey
                encrypted_watermark = encrypt_watermark(watermark_text, passkey)

                # Save the uploaded image
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                original_image = cv2.imread(filepath)
                if original_image is None:
                    return "Uploaded file is not a valid image.", 400

                # Apply denoising techniques
                median = median_denoise(original_image)
                gaussian = gaussian_denoise(original_image)
                laplacian = laplacian_denoise(original_image)

                # Save denoised images
                median_filename = 'median_' + filename
                median_filepath = os.path.join(app.config['UPLOAD_FOLDER'], median_filename)
                cv2.imwrite(median_filepath, median)

                gaussian_filename = 'gaussian_' + filename
                gaussian_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gaussian_filename)
                cv2.imwrite(gaussian_filepath, gaussian)

                laplacian_filename = 'laplacian_' + filename
                laplacian_filepath = os.path.join(app.config['UPLOAD_FOLDER'], laplacian_filename)
                cv2.imwrite(laplacian_filepath, laplacian)

                # Calculate SSIM scores
                ssim_median = calculate_ssim(original_image, median)
                ssim_gaussian = calculate_ssim(original_image, gaussian)
                ssim_laplacian = calculate_ssim(original_image, laplacian)

                denoise_results = {
                    'Median': {'image': median_filepath, 'ssim': ssim_median},
                    'Gaussian': {'image': gaussian_filepath, 'ssim': ssim_gaussian},
                    'Laplacian': {'image': laplacian_filepath, 'ssim': ssim_laplacian}
                }

                # Find the best method based on SSIM score
                best_method = max(denoise_results, key=lambda x: denoise_results[x]['ssim'])
                best_ssim = denoise_results[best_method]['ssim']

                # Set the SSIM threshold (e.g., 0.80 for 80%)
                threshold = 0.80

                # Logic to decide whether to use the best denoised image or revert to original
                if best_ssim < threshold:
                    # If the SSIM is below the threshold, revert to the original noisy image
                    print(f"SSIM is less than {threshold*100}%, reverting to original image.")
                    processed_image = original_image  # Keep the original noisy image
                    denoise_applied = False
                else:
                    # Otherwise, use the best denoised image
                    print(f"Using the best denoising method: {best_method} with SSIM: {best_ssim}")
                    processed_image = cv2.imread(denoise_results[best_method]['image'])  # Apply denoising
                    denoise_applied = True


                # Save processed image
                processed_filename = 'processed_' + filename
                processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                cv2.imwrite(processed_filepath, processed_image)

                # Embed the encrypted watermark into the image using LSB
                watermarked_lsb = lsb_watermark(processed_image, encrypted_watermark)
                lsb_filename = 'lsb_' + filename
                lsb_filepath = os.path.join(app.config['UPLOAD_FOLDER'], lsb_filename)
                cv2.imwrite(lsb_filepath, watermarked_lsb)

                # Embed the encrypted watermark into the image using Wavelet
                try:
                    watermarked_wavelet = wavelet_watermark(processed_image, encrypted_watermark)
                    wavelet_filename = 'wavelet_' + filename
                    wavelet_filepath = os.path.join(app.config['UPLOAD_FOLDER'], wavelet_filename)
                    cv2.imwrite(wavelet_filepath, watermarked_wavelet)
                except ValueError as ve:
                    return f"Error in watermarking: {ve}", 400

                # Extract watermarks for verification
                extracted_lsb = lsb_extract(watermarked_lsb, len(encrypted_watermark))
                extracted_wavelet = wavelet_extract(watermarked_wavelet, len(encrypted_watermark))

                # Decrypt extracted watermarks using the user-provided passkey
                try:
                    decrypted_lsb = decrypt_watermark(extracted_lsb, passkey)
                except InvalidToken:
                    decrypted_lsb = ""

                try:
                    decrypted_wavelet = decrypt_watermark(extracted_wavelet, passkey)
                except InvalidToken:
                    decrypted_wavelet = ""
                print(f"Original Watermark: {watermark_text}")
                print(f"Decrypted LSB Watermark: {decrypted_lsb}")
                print(f"Decrypted Wavelet Watermark: {decrypted_wavelet}")
                print(f"PASS KEY: {passkey}")

                # Calculate BER and PSNR
                ber_lsb = calculate_ber(extracted_lsb, passkey)
                ber_wavelet = calculate_ber(extracted_wavelet, passkey)
                psnr_lsb = calculate_psnr(processed_image, watermarked_lsb)
                psnr_wavelet = calculate_psnr(processed_image, watermarked_wavelet)


                print(f"Calculate psnr: {ber_lsb}")
                print(f"Calculate psnr: {ber_wavelet}")


                # Determine the best watermarking method
                if psnr_lsb > psnr_wavelet:
                    best_watermarked_image = lsb_filepath
                    best_method_watermark = 'LSB'
                    best_psnr = psnr_lsb
                    best_ber = ber_lsb
                else:
                    best_watermarked_image = wavelet_filepath
                    best_method_watermark = 'Wavelet'
                    best_psnr = psnr_wavelet
                    best_ber = ber_wavelet

                watermark_results = {
                    'LSB': {
                        'image': lsb_filepath,
                        'psnr': psnr_lsb,
                        'ber': ber_lsb
                    },
                    'Wavelet': {
                        'image': wavelet_filepath,
                        'psnr': psnr_wavelet,
                        'ber': ber_wavelet
                    }
                }

                # Render the template with the results
                return render_template('index.html',
                                       passkey = passkey,
                                       watermark_results=watermark_results,
                                       original_image=filepath,
                                       denoise_results=denoise_results,
                                       denoise_applied=denoise_applied,
                                       best_denoise_method=best_method if denoise_applied else 'None',
                                       processed_image=processed_filepath,
                                       best_watermark_method=best_method_watermark,
                                       best_watermarked_image=best_watermarked_image,
                                       psnr_best=best_psnr,
                                       ber_best=best_ber)

            except Exception as e:
                return f"An error occurred during processing: {e}", 500

    return render_template('index.html')

@app.route('/extract', methods=['GET', 'POST'])
def extract_watermark():
    if request.method == 'POST':
        if 'watermarked_image' not in request.files:
            return render_template('extract.html', error="No file uploaded")
        
        file = request.files['watermarked_image']
        if file.filename == '':
            return render_template('extract.html', error="No file selected")
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                watermarked_image = cv2.imread(filepath)
                if watermarked_image is None:
                    return render_template('extract.html', error="Uploaded file is not a valid image")

                passkey = request.form.get('passkey', '').strip()
                if not passkey:
                    return render_template('extract.html', error="Passkey is required for extraction")

                method = request.form.get('method')
                if method not in ['LSB', 'Wavelet']:
                    return render_template('extract.html', error="Invalid watermarking method selected")

                max_watermark_length = 1024  # Adjust this as needed

                if method == 'LSB':
                    extracted_encrypted = lsb_extract(watermarked_image, max_watermark_length)
                elif method == 'Wavelet':
                    extracted_encrypted = wavelet_extract(watermarked_image, max_watermark_length)

                if not extracted_encrypted:
                    return render_template('extract.html', error="Failed to extract watermark from the image")

                # Print the extracted watermark for debugging purposes
                print(f"Extracted Encrypted Watermark: {extracted_encrypted}")

                # Attempt to decrypt the watermark
                try:
                    decrypted_watermark = decrypt_watermark(extracted_encrypted, passkey)
                    print(f"Decrypted Watermark: {decrypted_watermark}")
                    watermark_info = json.loads(decrypted_watermark)
                except InvalidToken:
                    return render_template('extract.html', error="Invalid passkey or corrupted watermark. Decryption failed.")
                except json.JSONDecodeError:
                    return render_template('extract.html', error="Extracted watermark is not valid JSON")
                except Exception as e:
                    return render_template('extract.html', error=f"An error occurred during decryption: {str(e)}")

                return render_template('extract.html',
                                       watermark_info=watermark_info,
                                       watermarked_image=filepath)
            except Exception as e:
                return render_template('extract.html', error=f"An unexpected error occurred: {str(e)}")

    return render_template('extract.html')
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)