#include "grpc++/create_channel.h"
#include "grpc++/security/credentials.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/model_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using tensorflow::serving::GetModelMetricsRequest;
using tensorflow::serving::GetModelMetricsResponse;
using tensorflow::serving::ModelService;
using tensorflow::serving::ModelVersionMetrics_MethodType;
using tensorflow::serving::ModelVersionMetrics_MethodType_CLASSIFY;
using tensorflow::serving::ModelVersionMetrics_MethodType_REGRESS;
using tensorflow::serving::ModelVersionMetrics_MethodType_PREDICT;

class ServingClient {
 public:
  ServingClient(std::shared_ptr<Channel> channel) :
      stub_(ModelService::NewStub(channel)) {}

  tensorflow::string ToStringFromProtoEnum(
          const ModelVersionMetrics_MethodType& method_type) {
    switch (method_type) {
      case ModelVersionMetrics_MethodType_CLASSIFY: {
        return "Classify";
      }
      case ModelVersionMetrics_MethodType_REGRESS: {
        return "Regress";
      }
      case ModelVersionMetrics_MethodType_PREDICT: {
        return "Predict";
      }
      default:
        return "Unknown";
    }
  }

  void callMetrics(const tensorflow::string& model_name) {
    GetModelMetricsRequest request;
    GetModelMetricsResponse response;
    ClientContext context;

    request.mutable_model_spec()->set_name(model_name);

    Status status = stub_->GetModelMetrics(&context, request, &response);
    if (status.ok()) {
      for (int i=0; i<response.model_version_metrics_size(); i++){
        const auto& info = response.model_version_metrics(i);
        std::cout << "Name " << model_name <<
                     " Version " << info.version() <<
                     " MethodType " << ServingClient::ToStringFromProtoEnum(info.method_type()) <<
                     " Request Count " << info.metrics().request_count() <<
                     " Error Count " << info.metrics().error_count() << std::endl;
      }
    } else {
      std::cout << "gRPC call return code: " << status.error_code() << ": "
                << status.error_message() << std::endl;
    }
  }
 private:
  std::unique_ptr<ModelService::Stub> stub_;
};

int main(int argc, char** argv) {
  tensorflow::string server_port = "localhost:9000";
  tensorflow::string model_name = "mnist";
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("server_port", &server_port,
                       "the IP and port of the server"),
      tensorflow::Flag("model_name", &model_name,
                       "name of model")
  };

  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cout << usage;
    return -1;
  }

  ServingClient guide(
      grpc::CreateChannel(server_port, grpc::InsecureChannelCredentials()));
  guide.callMetrics(model_name);
}