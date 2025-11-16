#!/usr/bin/env python3
"""
Test script for the trained fire-detection model.

Features:
- Load the saved Sequential model `fire_detection_model.keras`
- Run prediction on a single image or all images in a directory (batch)
- Produce a Grad-CAM heatmap from the last Conv2D layer and overlay it
- Compute a simple bounding box from the heatmap and save annotated images
- Print per-image prediction and a small batch summary (accuracy if labels inferred from folder)

Usage:
  python dataset/images/val/test.py path/to/image.jpg
  python dataset/images/val/test.py --dir dataset/images/val/Fire

"""
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image


MODEL_PATH = os.path.abspath(os.path.join(os.getcwd(), "fire_detection_model.keras"))


def get_img_array(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img)
    arr = arr.astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
    if last_conv_layer_name is None:
        raise ValueError("No Conv2D layer found in model for Grad-CAM")
    # Build a model that maps the input image to the activations
    # of the last conv layer as well as the model's predictions
    # Build a functional model by applying the Sequential model's layers
    # to a new Input; this avoids relying on model.inputs/outputs
    input_shape = model.input_shape[1:]
    input_tensor = tf.keras.Input(shape=input_shape)
    x = input_tensor
    conv_output_tensor = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output_tensor = x

    if conv_output_tensor is None:
        raise RuntimeError(f"Could not capture output of layer {last_conv_layer_name}")

    # x now points to model output when applying original layers to new input
    grad_model = tf.keras.models.Model(inputs=input_tensor, outputs=[conv_output_tensor, x])

    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.cast(img_tensor, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = 0 if int(predictions.shape[-1]) == 1 else tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    # Compute gradients of the loss w.r.t. the conv layer outputs
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients are None; cannot compute Grad-CAM")

    # Pool the gradients across the spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    # Weight the channels by corresponding pooled gradients
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(orig_img_path, heatmap, out_path, alpha=0.5):
    orig = Image.open(orig_img_path).convert('RGB')
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(orig.size, resample=Image.BILINEAR)

    heatmap_arr = np.array(heatmap_img).astype('float32') / 255.0
    # red overlay
    red = np.zeros((heatmap_arr.shape[0], heatmap_arr.shape[1], 3), dtype='uint8')
    red[..., 0] = np.uint8(255 * heatmap_arr)
    red_img = Image.fromarray(red)

    blended = Image.blend(orig, red_img, alpha=alpha)
    blended.save(out_path)
    return out_path


def annotate_bbox(orig_img_path, heatmap, out_path, threshold=0.35, line_width=4, color=(255, 0, 0)):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    orig = Image.open(orig_img_path).convert('RGB')
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(orig.size, resample=Image.BILINEAR)
    heatmap_arr = np.array(heatmap_img).astype('float32') / 255.0

    # Use a higher threshold for tighter bbox
    tight_threshold = max(0.5, threshold)
    mask = heatmap_arr >= tight_threshold
    coords = np.argwhere(mask)
    annotated = orig.copy()
    draw = ImageDraw.Draw(annotated)

    # Overlay jet colormap for heatmap
    jet = cm.get_cmap('jet')
    jet_heatmap = (jet(heatmap_arr)[:, :, :3] * 255).astype(np.uint8)
    jet_img = Image.fromarray(jet_heatmap)
    jet_img = jet_img.convert('RGBA')
    overlay = Image.blend(annotated.convert('RGBA'), jet_img, alpha=0.4)

    if coords.size == 0:
        # fallback to overlay if no area above threshold
        base, ext = os.path.splitext(out_path)
        overlay.save(base + '_overlay' + ext)
        return base + '_overlay' + ext

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    pad_x = int(0.01 * annotated.width)
    pad_y = int(0.01 * annotated.height)
    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(annotated.width - 1, x_max + pad_x)
    y1 = min(annotated.height - 1, y_max + pad_y)

    draw = ImageDraw.Draw(overlay)
    for i in range(line_width):
        draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=color)

    overlay = overlay.convert('RGB')
    overlay.save(out_path)
    return out_path


def predict_and_visualize(model, img_path, out_path=None, threshold=0.35):
    if out_path is None:
        base, ext = os.path.splitext(img_path)
        out_path = base + '_annotated' + ext

    img_array = get_img_array(img_path, target_size=(128, 128))
    pred = model.predict(img_array)
    prob = float(pred[0][0])

    heatmap = make_gradcam_heatmap(img_array, model)
    annotated_path = annotate_bbox(img_path, heatmap, out_path, threshold=threshold)

    # Print simple detection result
    detected = prob > 0.5
    if detected:
        print(f"Fire detected (prob={prob:.4f})")
    else:
        print(f"No fire detected (prob={prob:.4f})")

    return prob, annotated_path, detected


def iterate_dir(model, dir_path, out_dir=None, threshold=0.35):
    if out_dir is None:
        out_dir = os.path.join(dir_path, 'annotated')
    os.makedirs(out_dir, exist_ok=True)

    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = []
    for f in sorted(files):
        base = os.path.basename(f)
        out_path = os.path.join(out_dir, base)
        prob, annotated = predict_and_visualize(model, f, out_path, threshold=threshold)
        results.append((f, prob, annotated))
        print(f"{base}: prob={prob:.4f} -> {annotated}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default='dataset/images/val/Fire/F_0.jpg',
                        help='Image path or directory')
    parser.add_argument('--dir', action='store_true', help='Treat path as directory and process all images inside')
    parser.add_argument('--out', default=None, help='Output directory (for --dir) or output file path')
    parser.add_argument('--threshold', type=float, default=0.35, help='Heatmap threshold for bbox')
    parser.add_argument('--show', action='store_true', help='Open the annotated image with the default image viewer')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please train and save the model first.")
        sys.exit(1)

    model = load_model(MODEL_PATH)
    print('Model loaded:', MODEL_PATH)

    if args.dir:
        results = iterate_dir(model, args.path, out_dir=args.out, threshold=args.threshold)
        # If directory has subfolders like Fire/ Non_Fire, user can run twice per folder or point to each
        # Print a small summary
        total = len(results)
        if total > 0:
            avg_prob = sum(r[1] for r in results) / total
            print(f"Processed {total} images. Avg prob (fire) = {avg_prob:.4f}")
    else:
        out_path = args.out
        prob, annotated, detected = predict_and_visualize(model, args.path, out_path, threshold=args.threshold)
        print(f"{args.path}: prob={prob:.4f} -> {annotated}")
        if args.show:
            try:
                # Use PIL's show which will call the OS default image viewer
                Image.open(annotated).show()
            except Exception as e:
                print('Could not open image viewer:', e)


if __name__ == '__main__':
    main()
