// Copyright 2014 BVLC and contributors.
//
// This is a simple script that allows one to quickly finetune a network.
// Usage:
//    finetune_net solver_proto_file pretrained_net

#include <cuda_runtime.h>
#include "caffe/util/upgrade_proto.hpp"
#include <string>
#include <fstream>
#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: finetune_net solver_proto_file pretrained_net";
    return 1;
  }

 // LOG(INFO) << "Starting Optimization";
 // std::ofstream outfile;
 // outfile.open("weights1.txt",std::ios::app);
 // NetParameter param;
 // ReadNetParamsFromBinaryFileOrDie(string(argv[2]), &param);
 // int num_source_layers = param.layers_size();
 // 
 //  for (int i = 0; i < num_source_layers; ++i) {
	//   const LayerParameter& source_layer = param.layers(i);
	//outfile<< source_layer.name()<<"\n";
	//for (int j = 0; j < source_layer.blobs_size() ; j++)
	//{
	//  const BlobProto& proto = source_layer.blobs(j);
	//  int count = proto.num()*proto.channels()*proto.height()*proto.width();
	//  for (int k = 0; k <count; k++)
	//  {
	//    outfile<< proto.data(k)<<"\n";
	//  }
	//  outfile<<"\n"<<"\n"<<"\n"<<"\n";
	//}
 // }
 // outfile.close();





  SolverParameter solver_param;
  ReadProtoFromTextFileOrDie(argv[1], &solver_param);

  LOG(INFO) << "Starting Optimization";
  SGDSolver<float> solver(solver_param);
  LOG(INFO) << "Loading from " << argv[2];
  solver.net()->CopyTrainedLayersFrom(string(argv[2]));
  solver.Solve();
  LOG(INFO) << "Optimization Done.";

  return 0;
}
