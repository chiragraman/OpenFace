/**
* @Author: Chirag Raman <chirag>
* @Date:   2016-05-09T21:14:02-04:00
* @Email:  chirag.raman@gmail.com
* @Last modified by:   chirag
* @Last modified time: 2016-12-19T15:11:46-05:00
* @License: Copyright (C) 2016 Multicomp Lab. All rights reserved.
*/

#include <iostream>
#include <random>
#include <string>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libavutil/imgutils.h"
#include "libavdevice/avdevice.h"
#include "libswscale/swscale.h"
}

#include <zmq.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <InmindEmotionDetector.h>
#include "json_dist/json/json.h"


using namespace InmindDemo;

// Use this to run from the camera with a device id like "/dev/video0" instead
// of grabbing from an RTSP stream
#define CAMERA_TEST (0)

// Inmind sends pictures sideways over rtsp. This flag is used for correcting
// the orientation of decoded images
#define INMIND_RTSP_CORRECTION (1)

// Used for drawing the decoded image to screen
#define DISPLAY_FRAME (0)

/********
 * HELPERS
 *******/

static std::string AVStrError(int errnum) {
    char buf[128];
    av_strerror(errnum, buf, sizeof(buf));
    return std::string(buf);
}

/**
 * Copied from libav/cmdutils.c because unlike FFmpeg, Libav does not export
 * this function in the public API.
 */
std::string media_type_string(enum AVMediaType media_type) {
    switch (media_type) {
    case AVMEDIA_TYPE_VIDEO:      return "video";
    case AVMEDIA_TYPE_AUDIO:      return "audio";
    case AVMEDIA_TYPE_DATA:       return "data";
    case AVMEDIA_TYPE_SUBTITLE:   return "subtitle";
    case AVMEDIA_TYPE_ATTACHMENT: return "attachment";
    default:                      return "unknown";
    }
}


/********
 * STATIC VARIABLES
 *******/

static AVCodecContext *video_decode_context = NULL;
static AVFrame *frame = NULL;
static AVFrame *frame_rgb = NULL;
static int video_stream_index = -1;
static InmindEmotionDetector *emotion_detector = NULL;
static std::string PORT = "5556";

/********
 * INITIALISATION
 *******/

void initialize(std::string executable_path) {
    av_register_all();
    avcodec_register_all();
    avformat_network_init();
    avdevice_register_all();

    emotion_detector = new InmindEmotionDetector(executable_path);
}

void init_format_context(AVFormatContext *&context) {
    context = avformat_alloc_context();
    if (!context) {
        av_log(0, AV_LOG_ERROR, "Cannot allocate input format (Out of memory?)\n");
        exit(1);
    }
    context->flags |= AVFMT_FLAG_NONBLOCK;
}

void init_input_format(AVInputFormat *&input_format) {
    input_format = av_find_input_format("v4l2");
    if (!input_format) {
        av_log(0, AV_LOG_ERROR, "Cannot find input format\n");
        exit(1);
    }
}

void init_options(AVDictionary *&options) {
    options = NULL;
    av_dict_set(&options, "framerate", "30", 0);
    av_dict_set(&options, "input_format", "h264", 0);
    av_dict_set(&options, "video_size", "1920x1080", 0);
}


/********
 * DEVICE INTERACTION
 *******/

void open_input(AVInputFormat *&input_format, AVDictionary *&options,
            AVFormatContext *&format_context, char const *device_name) {
    //Format context
    init_format_context(format_context);

    input_format = NULL;
    options = NULL;

#if( CAMERA_TEST )

    //Input Format
    init_input_format(input_format);

    //Options
    init_options(options);

#endif

    // check video source
    int open_result =
    avformat_open_input(&format_context, device_name, input_format, &options);

    if( open_result != 0) {
        std::cout << "\nOops, couldn't open video source:" << std::endl <<
         " error code  - " << open_result << std::endl <<
         " description - " << AVStrError(open_result) <<
         std::endl;
        exit(1);
    } else {
        av_log(0, AV_LOG_INFO, "Successfully opened camera!\n");
    }
}

void retrieve_stream_info(AVFormatContext *format_context) {
    if (avformat_find_stream_info(format_context, NULL) < 0) {
       av_log(0, AV_LOG_ERROR, "Could not find stream information\n");
       exit(1);
   }
}


/********
 * DECODING SETUP
 *******/

int open_codec_context(int *stream_index, AVFormatContext *format_context,
                       enum AVMediaType type, const char *device_name) {
    int ret = 0;
    AVStream *stream;
    AVCodecContext *decoder_context = NULL;
    AVCodec *decoder = NULL;
    AVDictionary *options = NULL;

    ret = av_find_best_stream(format_context, type, -1, -1, NULL, 0);
    if (ret < 0) {
        std::cerr << "Could not find " << media_type_string(type) <<
                  " stream from device " << device_name << std::endl;
    } else {
        *stream_index = ret;
        stream = format_context->streams[*stream_index];

        //Find decoder for the stream
        decoder_context = stream->codec;
        decoder = avcodec_find_decoder(decoder_context->codec_id);
        if (!decoder) {
            std::cerr << "Could not find the " << media_type_string(type) <<
                      " codec" << std::endl;
            ret = AVERROR(EINVAL);
        } else {
            //Init the decoders, with or without reference counting
            av_dict_set(&options, "refcounted_frames", "1", 0);
            ret = avcodec_open2(decoder_context, decoder, &options);
            if (ret < 0) {
                std::cerr << "Could not open the" <<  media_type_string(type)
                          << " codec" << std::endl;
            }
        }
    }
    return ret;
}

void configure_sw_scale_context(SwsContext *&conversion_context,
                           int width,
                           int height,
                           AVPixelFormat input_format,
                           AVPixelFormat destination_format) {
    conversion_context = sws_getCachedContext(
        NULL, width, height, input_format, width, height,
        destination_format, SWS_BICUBIC, NULL, NULL, NULL);
}

int setup_frame(AVFrame *&frame) {
    int ret = 0;
    if (!(frame = av_frame_alloc())) {
        std::cerr << "Could not allocate frame" << std::endl;
        ret = -1;
    } else {
        // Set the fields of the given AVFrame to default values
        av_frame_unref(frame);
    }
    return ret;
}

int setup_rgb_frame(AVFrame *&frame, uint8_t *&buffer,
                    AVPixelFormat pixel_format, int width, int height) {
    int ret = setup_frame(frame);
    avpicture_fill((AVPicture *)frame, buffer, pixel_format, width, height);

    return ret;
}

/********
 * DECODE
 *******/

double get_F0() {
    const double lower_bound = 85;
    const double upper_bound = 180;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::random_device rand_dev;
    std::mt19937 rand_engine(rand_dev());
    return unif(rand_engine);
}

double get_energy() {
    const double lower_bound = 65;
    const double upper_bound = 76;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::random_device rand_dev;
    std::mt19937 rand_engine(rand_dev());
    return unif(rand_engine);
}


 template <typename Iterable>
 Json::Value iterable_to_json(Iterable const& container) {
     Json::Value v;
     for (auto&& element : container) {
         v.append(element);
      }
      return v;
 }

std::string response_message(const FrameData &data, double pts) {
    Json::Value root;

    root["timestamp"] = pts;
    root["success"] = data.success;

    Json::Value gaze;
    gaze["gaze_left"] = iterable_to_json(data.gaze_estimate[0]);
    gaze["gaze_right"] = iterable_to_json(data.gaze_estimate[1]);

    root["features"]["gaze"] = gaze;
    root["features"]["pose"] = iterable_to_json(data.pose_estimate);
    root["features"]["action_units"] = iterable_to_json(data.aus);

    root["features"]["F0"] = get_F0();
    root["features"]["energy"] = get_energy();

    root["emotions"]["happiness"] = data.emotions[0];
    root["emotions"]["sadness"] = data.emotions[1];
    root["emotions"]["surprise"] = data.emotions[2];
    root["emotions"]["fear"] = data.emotions[3];
    root["emotions"]["anger"] = data.emotions[4];
    root["emotions"]["disgust"] = data.emotions[5];
    root["emotions"]["confusion"] = data.emotions[6];

    Json::FastWriter fastwriter;
    std::string message = fastwriter.write(root);
    return message;
}


 int decode_packet(AVPacket packet,
                   SwsContext *&sws_context,
                   int *got_frame,
                   int *video_frame_count,
                   int height,
                   int cached,
                   zmq::socket_t &sender_socket){
     int ret = 0;
     int decoded = packet.size;

     *got_frame = 0;

     if (packet.stream_index == video_stream_index) {
         //Decode video frame
         ret = avcodec_decode_video2(video_decode_context,
                                     frame,
                                     got_frame,
                                     &packet);
         if (ret < 0) {
             //FFmpeg users should use av_err2str
             char errbuf[128];
             av_strerror(ret, errbuf, sizeof(errbuf));
             std::cerr << "Error decoding video frame " << errbuf << std::endl;
             return ret;
         }

         if (*got_frame) {
             double pts = av_frame_get_best_effort_timestamp(frame);
             sws_scale(
                 sws_context,
                 ((AVPicture*)frame)->data,
                 ((AVPicture*)frame)->linesize,
                 0,
                 height,
                 ((AVPicture *)frame_rgb)->data,
                 ((AVPicture *)frame_rgb)->linesize);

             cv::Mat image_mat(frame->height, frame->width, CV_8UC3,
                               frame_rgb->data[0]);
#if( INMIND_RTSP_CORRECTION )
             cv::transpose(image_mat, image_mat);
             cv::flip(image_mat, image_mat, 0);
#endif
#if( DISPLAY_FRAME )
             cv::imshow("RTSP image",image_mat);
             cv::waitKey(1);
#endif
             FrameData data = emotion_detector->process_frame(image_mat, 0);
             std::string response_string = response_message(data, pts);

             zmq::message_t response(response_string.size());
             memcpy(response.data(), response_string.data(),
                    response_string.size());

             sender_socket.send(response);
         }
     }

     if (*got_frame) {
         av_frame_unref(frame);
     }

     return decoded;
 }



/********
 * GRAB FRAMES
 *******/

int process_frames(AVFormatContext *context,
                   SwsContext *&sws_context,
                   int height, zmq::socket_t &sender_socket) {
    int ret = 0;
    int got_frame;
    int video_frame_count = 0;
    AVPacket packet;

    //Initialize packet, set data to NULL, let the demuxer fill it
    av_init_packet(&packet);
    packet.data = NULL;
    packet.size = 0;

    // read frames from the file
    for (;;) {
        AVPacket orig_packet;
        ret = av_read_frame(context, &packet);
        if (ret < 0) {
            if  (ret == AVERROR(EAGAIN)) {
                continue;
            } else {
                break;
            }
        }

        orig_packet = packet;
        do {
            ret = decode_packet(packet, sws_context,
                                &got_frame, &video_frame_count, height, 0,
                                sender_socket);
            if (ret < 0) {
                break;
            }
            packet.data += ret;
            packet.size -= ret;
        } while (packet.size > 0);
        av_free_packet(&orig_packet);
    }

    //Flush cached frames
    packet.data = NULL;
    packet.size = 0;
    do {
        decode_packet(packet, sws_context,
                      &got_frame, &video_frame_count, height, 1, sender_socket);
    } while (got_frame);

    av_log(0, AV_LOG_INFO, "Demuxing succeeded\n");
    return ret;
}


/********
 * CLEANUP
 *******/

void cleanup(AVCodecContext *decode_context,
             AVFormatContext *format_context,
             AVFrame *frame,
             AVFrame *frame_rgb,
             SwsContext *sws_context) {
    avcodec_close(decode_context);
    avformat_close_input(&format_context);
    av_frame_free(&frame);
    av_frame_free(&frame_rgb);
    sws_freeContext(sws_context);
    delete emotion_detector;
}


/********
 * MAIN
 *******/

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage : " << argv[0] << " input_device_path"
                  << std::endl;
        exit(1);
    }

    //Get executable name
    std::string executable_name(argv[0]);

    //Declare local variables
    int ret = 0;
    char const *device_name = argv[1];

    AVInputFormat *input_format = NULL;
    AVDictionary *options = NULL;
    AVFormatContext *format_context = NULL;
    AVStream *video_stream = NULL;
    struct SwsContext *sws_context = NULL;
    AVPixelFormat destination_format = AV_PIX_FMT_BGR24;

    //Initialize
    initialize(executable_name);

    //Setup ZMQ
    zmq::context_t context(1);
    zmq::socket_t sender_socket(context, ZMQ_PAIR);
    std::string address = "tcp://*:" + PORT;
    sender_socket.bind(address);

    //Start reading stream
    open_input(input_format, options, format_context, device_name);
    retrieve_stream_info(format_context);

    //Setup video decoding
    if (open_codec_context(&video_stream_index, format_context,
                           AVMEDIA_TYPE_VIDEO, device_name) >= 0) {
        video_stream = format_context->streams[video_stream_index];
        if (!video_stream) {
            std::cerr << "Could not find video stream in the input, aborting"
                      << std::endl;
        }

        video_decode_context = video_stream->codec;

        int width = video_decode_context->width;
        int height = video_decode_context->height;

        configure_sw_scale_context(sws_context, width, height,
                                   video_decode_context->pix_fmt,
                                   destination_format);

        //Setup frames
        //Both methods should return 0 if successul
        uint8_t *buffer;
        int numBytes;
        numBytes  = avpicture_get_size(destination_format, width, height);
        buffer = (uint8_t *) av_malloc(numBytes*sizeof(uint8_t));

        int frame_success = (setup_frame(frame) == 0) &&
            (setup_rgb_frame(frame_rgb, buffer, destination_format,
                             width, height) == 0);

        //Abort if any step of the setup failed
        if (!video_stream || !frame_success) {
            cleanup(video_decode_context,
                    format_context,
                    frame, frame_rgb, sws_context);
            exit(1);
        } else {
            av_log(0, AV_LOG_INFO, "Demuxing video from %s", device_name);
        }
    }

    //Dump input information to stderr
    av_dump_format(format_context, 0, device_name, 0);
    std::cout<<std::endl;

    //Start processing frames
    ret = process_frames(format_context, sws_context,
                         video_decode_context->height, sender_socket);

    return ret < 0;
}
