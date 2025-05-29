# https://drive.google.com/file/d/1-fihBHwcU0JX81zZJPFYXCnuO_l1pVND/view?usp=share_link - model_final.pth
# https://drive.google.com/file/d/1mKfPO-yQ8iwsqC-rAlVofwSRJi52nKzK/view?usp=share_link - model_simplified.onnx
# https://drive.google.com/file/d/109MUa5ukzHup6Xdu9Rj_iCoF1rSp-VaV/view?usp=share_link - model.onnx
import gdown


model_final_url = 'https://drive.google.com/uc?id=1-fihBHwcU0JX81zZJPFYXCnuO_l1pVND'
model_simplified_url = 'https://drive.google.com/uc?id=1mKfPO-yQ8iwsqC-rAlVofwSRJi52nKzK'
model_url = 'https://drive.google.com/uc?id=109MUa5ukzHup6Xdu9Rj_iCoF1rSp-VaV'

gdown.download(model_final_url, 'model_final.pth', quiet=False)
gdown.download(model_simplified_url, 'model_simplified.onnx', quiet=False)
gdown.download(model_url, 'model.onnx', quiet=False)
print('Weights downloaded successfully.')