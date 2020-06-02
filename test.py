import cv2
import tensorflow as tf 
from model_aon import get_init_op
import os

flags = tf.app.flags
flags.DEFINE_string('exp_dir', 'exp_log', '')
flags.DEFINE_string('mode', 'tags', '')
flags.DEFINE_string('image_path', 'C://Users/Delphy/Downloads/mjsynth.tar/mjsynth/mnt/ramdisk/max/90kDICT32px/1388/2/21_glans_32655.jpg', '')
flags.DEFINE_string('tags_file', 'C://Users/Delphy/Downloads/tags_file.txt', '')
FLAGS = flags.FLAGS


def load_image(image_path):
  image = cv2.imread(image_path)
  image = cv2.resize(image, (100, 100))
  image = image / 255.0
  return image

def test_single_picture():
  save_path = tf.train.latest_checkpoint(FLAGS.exp_dir)
  meta_file_path = save_path + '.meta'
  tf.compat.v1.reset_default_graph()
  saver = tf.compat.v1.train.import_meta_graph(meta_file_path)

  sess = tf.compat.v1.Session()
  sess.run(get_init_op())
  saver.restore(sess, save_path=save_path)  # restore sess

  graph = tf.compat.v1.get_default_graph()
  global_step = graph.get_tensor_by_name('global_step:0')
  image_placeholder = graph.get_tensor_by_name('input/Placeholder:0')
  output_eval_text_tensor = graph.get_tensor_by_name('attention_decoder/ReduceJoin_1/ReduceJoin:0')
  print('Restore graph from meta file {}'.format(meta_file_path))
  print('Restore model from {} successful, step {}'.format(save_path, sess.run(global_step)))
  if FLAGS.mode == 'single':
    pred_text = sess.run(output_eval_text_tensor, feed_dict={
      image_placeholder: load_image(FLAGS.image_path).reshape([1, 100, 100, 3])
    })

    print(pred_text[0].decode())
  elif FLAGS.mode == 'tags':
    num_total = 0
    num_correct = 0
    with open(FLAGS.tags_file) as fo:
      for line in fo:
        try:
          image_path, gt = line.strip().split(' ')
          image = load_image(image_path)
        except Exception as e:
          print(e, image_path)
          continue
        pred_text = sess.run(output_eval_text_tensor, feed_dict={
          image_placeholder: image.reshape([1, 100, 100, 3])
        })
        print('{} ==> {}'.format(gt.lower(), pred_text[0].decode()))
        num_total += 1
        num_correct += (gt.lower() == pred_text[0].decode())
    print('Accu: {}/{}={}'.format(num_correct, num_total, num_correct/num_total))
    print('{}'.format(save_path))
  else:
    raise ValueError('Unsupported mode: {}'.format(FLAGS.mode))
  sess.close()


if __name__ == '__main__':
  test_single_picture()
