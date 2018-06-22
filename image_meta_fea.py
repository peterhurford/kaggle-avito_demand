
def transform_image(img):
  yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  rgb_r = img[:, :, 0]
  rgb_g = img[:, :, 1]
  rgb_b = img[:, :, 2]
  yuv_y = yuv[:, :, 0]
  hsv_s = hsv[:, :, 1]
  hsv_v = hsv[:, :, 2]
  hls_l = hls[:, :, 1]

  res = []

  # Lightness
  res.append(np.mean(yuv_y))
  res.append(np.std(yuv_y))

  res.append(np.mean(hls_l))
  res.append(np.std(hls_l))

  # Saturation
  res.append(np.mean(hsv_s))
  res.append(np.std(hsv_s))

  # Colorfulness
  rgb_rg = rgb_r - rgb_g
  rgb_yb = (rgb_r + rgb_g) / 2 - rgb_b
  colorful = np.sqrt(np.var(rgb_rg) + np.var(rgb_yb)) + \
             0.3 * np.sqrt(np.mean(rgb_rg) ** 2 + np.mean(rgb_yb) ** 2)
  res.append(colorful)

  # Gray
  res.append(np.std(gray))

  # Color
  res.append(np.mean(rgb_r))
  res.append(np.mean(rgb_g))
  res.append(np.mean(rgb_b))

  # Shape
  shape = img.shape
  res.append(shape[0])
  res.append(shape[1])
  res.append(shape[0] * shape[1])

  # Blurrness
  res.append(cv2.Laplacian(gray, cv2.CV_64F).var())
  return res
