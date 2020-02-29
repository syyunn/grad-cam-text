import tensorflow as tf
import os
from model import Model
from dataset import SST, Word2vecEnWordEmbedder

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


data = SST(Word2vecEnWordEmbedder)

model = Model()

max_article_length = 500
classifier = tf.estimator.Estimator(
    model_fn=model.build,
    config=tf.estimator.RunConfig(session_config=config),
    params={
        "feature_columns": [tf.feature_column.numeric_column(key="x")],
        "kernels": [(3, 512), (4, 512), (5, 512)],
        "num_classes": 2,
        "max_article_length": max_article_length,
    },
)

checkpoint_path = "./ckpt/0229_162533/model.ckpt-0"  # shall be customized

sentences = [
    "the movie exists for its soccer action and its fine acting .",
    "the thrill is -lrb- long -rrb- gone .",
    "now it 's just tired .",
    "the cast is very excellent and relaxed .",
]

pred_val = classifier.predict(
    input_fn=lambda: data.predict_input_fn(sentences, padded_size=max_article_length),
    checkpoint_path=checkpoint_path,
)

for i, _val in enumerate(pred_val):
    pred_idx = _val["predict_index"][0]
    vec = _val["grad_cam"][pred_idx][:17]
    pred_text = "Negative" if pred_idx == 0 else "Positive"
    # _plot_score(vec=vec, pred_text=pred_text, xticks=_get_text_xticks(sentences[i]))
    print(vec, pred_text)


if __name__ == "__main__":
    pass
