/**
* @Author: Chirag Raman <chirag>
* @Date:   2016-07-29T14:35:48-04:00
* @Email:  chirag.raman@gmail.com
* @Last modified by:   chirag
* @Last modified time: 2016-08-02T10:54:51-04:00
* @License: Copyright (C) 2016 Multicomp Lab. All rights reserved.
*/

#ifndef EXE_INMIND_DEMO_MULTISENSE_SERVER_H_
#define EXE_INMIND_DEMO_MULTISENSE_SERVER_H_

#include <thread>
#include <unordered_map>
#include <zmq.hpp>

class MultisenseServer {
 public:
     MultisenseServer();
     virtual ~MultisenseServer ();
 private:
     struct Client {
         std::string device_id;
         std::string url;
         std::string type_request;
         int port;
     };
     std::unordered_map<std::string,Client> phone_clients_;
     std::unordered_map<std::string,Pairserver> paired_servers_;


};

#endif /* end of include guard: EXE_INMIND_DEMO_MULTISENSE_SERVER_H_ */
