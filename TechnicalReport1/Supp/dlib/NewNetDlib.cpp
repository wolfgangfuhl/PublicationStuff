
#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>



using namespace std;
using namespace dlib;



void load_dataset_CIFAR100(std::string path,
	std::vector<matrix<dlib::rgb_pixel>>& training_images,
	std::vector<unsigned long>& training_labels,
	std::vector<matrix<dlib::rgb_pixel>>& testing_images,
	std::vector<unsigned long>& testing_labels) {

	training_images.clear();
	training_labels.clear();
	testing_images.clear();
	testing_labels.clear();

	dlib::rand rnd(time(0));

	std::string basepath = path + "/cifar100/";


	int min_label = 1000;
	int max_label = 0;

	std::string all_files[2];
	all_files[0] = "train.bin";
	all_files[1] = "test.bin";


	for (int afi = 0; afi < 2; afi++) {

		std::cout << "Loading data: " << training_images.size() + testing_images.size() << std::endl;
		std::cout << "Loading file: " << all_files[afi] << std::endl;

		std::string filename = all_files[afi];


		std::ifstream input(basepath + filename, std::ios::binary);
		std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});


		unsigned int akidx = 0;
		while (akidx < buffer.size()) {

			unsigned int aklabel = buffer[akidx++];
			aklabel = buffer[akidx++];



			matrix<dlib::rgb_pixel> akD;
			akD.set_size(40, 40);


			for (int y_i = 0; y_i < 40; y_i++) {
				for (int x_i = 0; x_i < 40; x_i++) {
					akD(y_i, x_i).red = 0;
					akD(y_i, x_i).green = 0;
					akD(y_i, x_i).blue = 0;
				}
			}


			for (int ch_i = 0; ch_i < 3; ch_i++) {
				for (int y_i = 0; y_i < 32; y_i++) {
					for (int x_i = 0; x_i < 32; x_i++) {

						if (ch_i == 0)
							akD(y_i + 4, x_i + 4).red = buffer[akidx++];
						if (ch_i == 1)
							akD(y_i + 4, x_i + 4).green = buffer[akidx++];
						if (ch_i == 2)
							akD(y_i + 4, x_i + 4).blue = buffer[akidx++];



					}
				}
			}



			if (afi < 1) {
				training_images.push_back(akD);
				training_labels.push_back(aklabel);


			}
			else {

				testing_labels.push_back(aklabel);
				testing_images.push_back(akD);
			}


		}

	}




}








template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;


template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;


template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;



template <int N, typename SUBNET> using res = relu<residual<block, N, bn_con, SUBNET>>;//bn_con
template <int N, typename SUBNET> using res_down = relu<residual_down<block, N, bn_con, SUBNET>>;
template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;


template <typename SUBNET> using level1 = res<2048, res_down<2048, SUBNET>>;
template <typename SUBNET> using level2 = res<1024, res_down<1024, SUBNET>>;
template <typename SUBNET> using level3 = res<512, res_down<512, SUBNET>>;
template <typename SUBNET> using level4 = res<256, SUBNET>;

template <typename SUBNET> using alevel1 = ares<2048, ares_down<2048, SUBNET>>;
template <typename SUBNET> using alevel2 = ares<1024, ares_down<1024, SUBNET>>;
template <typename SUBNET> using alevel3 = ares<512, ares_down<512, SUBNET>>;
template <typename SUBNET> using alevel4 = ares<256, SUBNET>;

template <typename SUBNET>
using withBN = concat5<tag7, tag8, tag9, tag10, tag11,
	tag7< avg_pool_everything< level1 < skip4<
	tag8< avg_pool_everything< tag4<level2 < skip5<
	tag9< avg_pool_everything< tag5<level3 < skip6<
	tag10< avg_pool_everything< tag6<level4 < skip12<
	tag11< avg_pool_everything< tag12<relu<bn_con<con<128, 5, 5, 1, 1,
	SUBNET>>>>>>>>>
	>>>>>>>>>>>>>>>>>;

template <typename SUBNET>
using noBN = concat5<tag7, tag8, tag9, tag10, tag11,
	tag7< avg_pool_everything< alevel1 < skip4<
	tag8< avg_pool_everything< tag4<alevel2 < skip5<
	tag9< avg_pool_everything< tag5<alevel3 < skip6<
	tag10< avg_pool_everything< tag6<alevel4 < skip12<
	tag11< avg_pool_everything< tag12<relu<affine<con<128, 5, 5, 1, 1,
	SUBNET>>>>>>>>>
	>>>>>>>>>>>>>>>>>;

using anet_type = loss_multiclass_log<fc<100,
	withBN<
	input<matrix<dlib::rgb_pixel>>
	>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type_ev = loss_multiclass_log<fc<100,
	noBN<
	input<matrix<dlib::rgb_pixel>>
	>>>;

std::string name_best = "cifar_100_best_cnn.dat";




int main(int argc, char** argv) try
{

	int device_id = 0;

	dlib::cuda::set_device(device_id);




	for (int TOTALRUN = 0; TOTALRUN < 5; TOTALRUN++) {
		
		float start_LR = 0.01;
		int mini_batch_sz = 10;
		float error_best = 0.1;
		
		anet_type net;

		anet_type_ev evnet;

		int AK_LEARNING_RATE = 0;



		std::vector<matrix<dlib::rgb_pixel>> training_images;
		std::vector<unsigned long>         training_labels;


		std::vector<matrix<dlib::rgb_pixel>> testing_images;
		std::vector<unsigned long>         testing_labels;

		load_dataset_CIFAR100("E:/datasetsImage", training_images, training_labels, testing_images, testing_labels);


		cout << "training_images: " << training_images.size() << endl;
		cout << "training_labels: " << training_labels.size() << endl;
		cout << "testing_images: " << testing_images.size() << endl;
		cout << "testing_labels: " << testing_labels.size() << endl;



		ofstream myfile_loss;
		myfile_loss.open(name_best + "_" + std::to_string(TOTALRUN) + "_" + std::to_string(AK_LEARNING_RATE) + "_loss.txt");
		myfile_loss << "Loss;ACC" << std::endl;

		ofstream myfile;


		







		dnn_trainer<anet_type, sgd> trainer(net, sgd(0.0005, 0.9), { device_id });
		trainer.set_learning_rate(start_LR);
		trainer.set_min_learning_rate(0.000000000001);
		trainer.set_mini_batch_size(mini_batch_sz);
		trainer.be_verbose();
		trainer.set_iterations_without_progress_threshold(2000000);
		trainer.set_max_num_epochs(1000000);


		std::vector<matrix<dlib::rgb_pixel>> mini_batch_samples;
		std::vector<unsigned long> mini_batch_labels;
		dlib::rand rnd;


		_int64 cnt = 1;
		int epoch_cnt = 0;


		while (cnt < _int64(training_images.size() * 1000))
		{


			if (cnt % (training_images.size() / mini_batch_sz) != 0) {

				mini_batch_samples.clear();
				mini_batch_labels.clear();


				if (rnd.get_double_in_range(0.0, 1.0) >= 0.5 && mini_batch_sz >= 100) {

					for (int wri = 0; wri < mini_batch_sz; wri++) {
						int akv = rnd.get_integer(training_images.size());

						while (wri % 100 != training_labels[akv]) {
							akv = rnd.get_integer(training_images.size());
						}

						if (mini_batch_samples.size() < mini_batch_sz) {




							matrix<dlib::rgb_pixel> akD = training_images[akv];
							matrix<dlib::rgb_pixel> akCropp;
							akCropp.set_size(32, 32);

							int rnd_x = rnd.get_integer_in_range(0, 8);
							int rnd_y = rnd.get_integer_in_range(0, 8);

							for (int y_i = 0; y_i < 32; y_i++) {
								for (int x_i = 0; x_i < 32; x_i++) {
									akCropp(y_i, x_i).red = akD(y_i + rnd_y, x_i + rnd_x).red;
									akCropp(y_i, x_i).green = akD(y_i + rnd_y, x_i + rnd_x).green;
									akCropp(y_i, x_i).blue = akD(y_i + rnd_y, x_i + rnd_x).blue;
								}
							}




							mini_batch_samples.push_back(akCropp);
							mini_batch_labels.push_back(training_labels[akv]);
						}


					}
				}
				else {

					for (int wri = 0; wri < mini_batch_sz; wri++) {
						int akv = rnd.get_integer(training_images.size());

						if (mini_batch_samples.size() < mini_batch_sz) {




							matrix<dlib::rgb_pixel> akD = training_images[akv];
							matrix<dlib::rgb_pixel> akCropp;
							akCropp.set_size(32, 32);

							int rnd_x = rnd.get_integer_in_range(0, 8);
							int rnd_y = rnd.get_integer_in_range(0, 8);

							for (int y_i = 0; y_i < 32; y_i++) {
								for (int x_i = 0; x_i < 32; x_i++) {
									akCropp(y_i, x_i).red = akD(y_i + rnd_y, x_i + rnd_x).red;
									akCropp(y_i, x_i).green = akD(y_i + rnd_y, x_i + rnd_x).green;
									akCropp(y_i, x_i).blue = akD(y_i + rnd_y, x_i + rnd_x).blue;
								}
							}







							mini_batch_samples.push_back(akCropp);
							mini_batch_labels.push_back(training_labels[akv]);
						}


					}


				}


				for (int wri = 0; wri < mini_batch_sz; wri++) {


					if (rnd.get_integer_in_range(0, 101) > 50) {
						matrix<dlib::rgb_pixel> akmat = dlib::fliplr(mini_batch_samples[wri]);
						mini_batch_samples[wri] = akmat;
					}

				}

				trainer.train_one_step(mini_batch_samples.cbegin(), mini_batch_samples.cend(), mini_batch_labels.cbegin());

			}
			else {
				//eval

				cnt = 0;
				epoch_cnt++;

				trainer.get_net(dlib::force_flush_to_disk::no);
				serialize(name_best + "_help.dat") << net;

				deserialize(name_best + "_help.dat") >> evnet;


				myfile.open(name_best + "_" + std::to_string(TOTALRUN) + "_" + std::to_string(AK_LEARNING_RATE) + "_clean.txt");
				myfile << "Pred;GT" << std::endl;

				int num_right = 0;
				int num_wrong = 0;
				float result = 0;
				for (size_t i = 0; i < testing_images.size(); ++i) {



					matrix<dlib::rgb_pixel> akD = testing_images[i];
					matrix<dlib::rgb_pixel> akCropp;
					akCropp.set_size(32, 32);

					int rnd_x = 4;
					int rnd_y = 4;

					for (int y_i = 0; y_i < 32; y_i++) {
						for (int x_i = 0; x_i < 32; x_i++) {
							akCropp(y_i, x_i).red = akD(y_i + rnd_y, x_i + rnd_x).red;
							akCropp(y_i, x_i).green = akD(y_i + rnd_y, x_i + rnd_x).green;
							akCropp(y_i, x_i).blue = akD(y_i + rnd_y, x_i + rnd_x).blue;
						}
					}



					unsigned long predicted_labels = evnet.process(akCropp);

					if (predicted_labels == testing_labels[i]) {
						++num_right;
					}
					else {
						++num_wrong;
					}

					myfile << predicted_labels << ";" << testing_labels[i] << std::endl;
				}

				result = num_right / (double)(num_right + num_wrong);
				cout << "testing num_right: " << num_right << endl;
				cout << "testing num_wrong: " << num_wrong << endl;
				cout << "testing accuracy:  " << result << " | " << error_best << endl;

				myfile << result << ";" << error_best << std::endl;
				myfile.close();

				myfile_loss << trainer.get_average_loss() << ";" << result << ";" << error_best << std::endl;

				if (result > error_best) {
					error_best = result;
					serialize(name_best) << net;
				}


				if (epoch_cnt % 100 == 0) {
					trainer.set_learning_rate(trainer.get_learning_rate()*0.1);

					AK_LEARNING_RATE++;

				}
				if (epoch_cnt == 400) {
					break;
					cnt = _int64(training_images.size()) * _int64(1000);
				}



			}


			cnt++;

		}


		myfile_loss.close();
	}


}
catch (std::exception& e)
{
	cout << e.what() << endl;
}

