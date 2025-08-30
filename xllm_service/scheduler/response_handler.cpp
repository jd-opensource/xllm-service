/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "response_handler.h"

namespace xllm_service {

bool ResponseHandler::send_delta_to_client(
    std::shared_ptr<ChatCallData> call_data,
    std::unordered_set<size_t>* first_message_sent,
    bool include_usage,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model,
    const llm::RequestOutput& output) {
  auto& response = call_data->response();

  // send delta to client
  for (const auto& seq_output : output.outputs) {
    const auto& index = seq_output.index;

    // send first chunk with role as assistant
    if (first_message_sent->find(index) == first_message_sent->end()) {
      response.Clear();
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      auto* message = choice->mutable_delta();
      message->set_role("assistant");
      message->set_content("");
      // update first_message_sent
      first_message_sent->insert(index);
      if (!call_data->write(response)) {
        return false;
      }
    }

    // send chunk with delta message
    if (!seq_output.text.empty()) {
      response.Clear();
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);

      // set_logprobs
      if (seq_output.logprobs.has_value() &&
          !seq_output.logprobs.value().empty()) {
        auto* proto_logprobs = choice->mutable_logprobs();
        proto_logprobs->mutable_content()->Reserve(
            seq_output.logprobs.value().size());
        for (const auto& logprob : seq_output.logprobs.value()) {
          auto* logprob_proto = proto_logprobs->add_content();
          logprob_proto->set_token(logprob.token);
          logprob_proto->set_token_id(logprob.token_id);
          logprob_proto->set_logprob(logprob.logprob);

          if (logprob.top_logprobs.has_value()) {
            for (const auto& top_logprob : logprob.top_logprobs.value()) {
              auto* top_logprob_proto = logprob_proto->add_top_logprobs();
              top_logprob_proto->set_token(top_logprob.token);
              top_logprob_proto->set_token_id(top_logprob.token_id);
              top_logprob_proto->set_logprob(top_logprob.logprob);
            }
          }
        }
      }

      auto* message = choice->mutable_delta();
      message->set_content(seq_output.text);
      if (!call_data->write(response)) {
        return false;
      }
    }

    // send a separate chunk with finish reason
    if (seq_output.finish_reason.has_value()) {
      response.Clear();
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      choice->mutable_delta();
      choice->set_finish_reason(seq_output.finish_reason.value());
      if (!call_data->write(response)) {
        return false;
      }
    }
  }

  // send additional chunk for usage statistics
  if (include_usage && output.usage.has_value()) {
    response.Clear();
    const auto& usage = output.usage.value();
    response.set_object("chat.completion.chunk");
    response.set_id(request_id);
    response.set_created(created_time);
    response.set_model(model);
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(
        static_cast<int32_t>(usage.num_prompt_tokens));
    proto_usage->set_completion_tokens(
        static_cast<int32_t>(usage.num_generated_tokens));
    proto_usage->set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
    if (!call_data->write(response)) {
      return false;
    }
  }

  if (output.finished) {
    response.Clear();
    return call_data->finish();
  }
  return true;
}

bool ResponseHandler::send_delta_to_client(
    std::shared_ptr<CompletionCallData> call_data,
    bool include_usage,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model,
    const llm::RequestOutput& output) {
  auto& response = call_data->response();

  for (const auto& seq_output : output.outputs) {
    // send chunk with delta message
    if (!seq_output.text.empty()) {
      response.Clear();
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text(seq_output.text);

      // set_logprobs
      if (seq_output.logprobs.has_value() &&
          !seq_output.logprobs.value().empty()) {
        auto* proto_logprobs = choice->mutable_logprobs();
        for (const auto& logprob : seq_output.logprobs.value()) {
          proto_logprobs->add_tokens(logprob.token);
          proto_logprobs->add_token_ids(logprob.token_id);
          proto_logprobs->add_token_logprobs(logprob.logprob);
        }
      }

      if (!call_data->write(response)) {
        return false;
      }
    }

    // send a separate chunk with finish reason
    if (seq_output.finish_reason.has_value()) {
      response.Clear();
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text("");
      choice->set_finish_reason(seq_output.finish_reason.value());
      if (!call_data->write(response)) {
        return false;
      }
    }
  }

  // send additional chunk for usage statistics
  if (include_usage && output.usage.has_value()) {
    const auto& usage = output.usage.value();
    response.Clear();
    response.set_object("text_completion");
    response.set_id(request_id);
    response.set_created(created_time);
    response.set_model(model);
    response.mutable_choices();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(
        static_cast<int32_t>(usage.num_prompt_tokens));
    proto_usage->set_completion_tokens(
        static_cast<int32_t>(usage.num_generated_tokens));
    proto_usage->set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
    if (!call_data->write(response)) {
      return false;
    }
  }

  if (output.finished) {
    response.Clear();
    // TODO: convert status to grpc status code
    return call_data->finish();
  }
  return true;
}

bool ResponseHandler::send_result_to_client(
    std::shared_ptr<ChatCallData> call_data,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model,
    const llm::RequestOutput& req_output) {
  auto& response = call_data->response();
  response.set_object("chat.completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  response.mutable_choices()->Reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    // add choices into response
    auto* choice = response.add_choices();
    choice->set_index(output.index);

    // set_logprobs
    if (output.logprobs.has_value() && !output.logprobs.value().empty()) {
      auto* proto_logprobs = choice->mutable_logprobs();
      proto_logprobs->mutable_content()->Reserve(
          output.logprobs.value().size());
      for (const auto& logprob : output.logprobs.value()) {
        auto* logprob_proto = proto_logprobs->add_content();
        logprob_proto->set_token(logprob.token);
        logprob_proto->set_token_id(logprob.token_id);
        logprob_proto->set_logprob(logprob.logprob);

        if (logprob.top_logprobs.has_value()) {
          for (const auto& top_logprob : logprob.top_logprobs.value()) {
            auto* top_logprob_proto = logprob_proto->add_top_logprobs();
            top_logprob_proto->set_token(top_logprob.token);
            top_logprob_proto->set_token_id(top_logprob.token_id);
            top_logprob_proto->set_logprob(top_logprob.logprob);
          }
        }
      }
    }

    auto* message = choice->mutable_message();
    message->set_role("assistant");
    message->set_content(output.text);
    if (output.finish_reason.has_value()) {
      choice->set_finish_reason(output.finish_reason.value());
    }
  }

  // add usage statistics
  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(
        static_cast<int32_t>(usage.num_prompt_tokens));
    proto_usage->set_completion_tokens(
        static_cast<int32_t>(usage.num_generated_tokens));
    proto_usage->set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
  }

  return call_data->write_and_finish(response);
}

bool ResponseHandler::send_result_to_client(
    std::shared_ptr<CompletionCallData> call_data,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model,
    const llm::RequestOutput& req_output) {
  auto& response = call_data->response();
  response.set_object("text_completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  // add choices into response
  response.mutable_choices()->Reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    choice->set_text(output.text);

    // set_logprobs
    if (output.logprobs.has_value() && !output.logprobs.value().empty()) {
      auto* proto_logprobs = choice->mutable_logprobs();
      for (const auto& logprob : output.logprobs.value()) {
        proto_logprobs->add_tokens(logprob.token);
        proto_logprobs->add_token_ids(logprob.token_id);
        proto_logprobs->add_token_logprobs(logprob.logprob);
      }
    }

    if (output.finish_reason.has_value()) {
      choice->set_finish_reason(output.finish_reason.value());
    }
  }

  // add usage statistics
  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(
        static_cast<int32_t>(usage.num_prompt_tokens));
    proto_usage->set_completion_tokens(
        static_cast<int32_t>(usage.num_generated_tokens));
    proto_usage->set_total_tokens(static_cast<int32_t>(usage.num_total_tokens));
  }

  return call_data->write_and_finish(response);
}

}  // namespace xllm_service
