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

#include <optional>

#include "scheduler/xllm_chat_parse_bridge.h"
#include "xllm/xllm/api_service/stream_output_parser.h"
#include "xllm/xllm/api_service/utils.h"
#include "xllm/xllm/function_call/function_call_parser.h"

namespace xllm_service {
namespace {

void set_logprobs(xllm::proto::ChatChoice* choice,
                  const std::optional<std::vector<llm::LogProb>>& logprobs) {
  if (!logprobs.has_value() || logprobs->empty()) {
    return;
  }

  auto* proto_logprobs = choice->mutable_logprobs();
  proto_logprobs->mutable_content()->Reserve(logprobs->size());
  for (const auto& logprob : *logprobs) {
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

bool send_normal_text_chunk(std::shared_ptr<ChatCallData> call_data,
                            size_t index,
                            const std::string& content,
                            const std::string& request_id,
                            int64_t created_time,
                            const std::string& model) {
  if (content.empty()) {
    return true;
  }

  auto& response = call_data->response();
  response.Clear();
  response.set_object("chat.completion.chunk");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);
  auto* choice = response.add_choices();
  choice->set_index(index);
  auto* delta = choice->mutable_delta();
  delta->set_content(content);
  return call_data->write(response);
}

bool send_reasoning_text_chunk(std::shared_ptr<ChatCallData> call_data,
                               size_t index,
                               const std::string& content,
                               const std::string& request_id,
                               int64_t created_time,
                               const std::string& model) {
  if (content.empty()) {
    return true;
  }

  auto& response = call_data->response();
  response.Clear();
  response.set_object("chat.completion.chunk");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);
  auto* choice = response.add_choices();
  choice->set_index(index);
  auto* message = choice->mutable_delta();
  message->set_reasoning_content(content);
  return call_data->write(response);
}

bool send_tool_call_chunk(std::shared_ptr<ChatCallData> call_data,
                          size_t index,
                          const std::string& tool_call_id,
                          const std::string& function_name,
                          const std::string& arguments,
                          int tool_index,
                          const std::string& request_id,
                          int64_t created_time,
                          const std::string& model) {
  auto& response = call_data->response();
  response.Clear();
  response.set_object("chat.completion.chunk");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  auto* choice = response.add_choices();
  choice->set_index(index);
  auto* delta = choice->mutable_delta();
  auto* tool_call = delta->add_tool_calls();
  if (!tool_call_id.empty()) {
    tool_call->set_id(tool_call_id);
  }
  tool_call->set_index(tool_index);
  tool_call->set_type("function");

  auto* function = tool_call->mutable_function();
  if (!function_name.empty()) {
    function->set_name(function_name);
  }
  if (!arguments.empty()) {
    function->set_arguments(arguments);
  }

  return call_data->write(response);
}

bool process_tool_call_stream(
    std::shared_ptr<ChatCallData> call_data,
    std::shared_ptr<xllm::StreamOutputParser> stream_parser,
    size_t index,
    const std::string& delta,
    const std::string& request_id,
    int64_t created_time,
    const std::string& model) {
  auto* parser = stream_parser->get_tool_call_parser(index);
  if (!parser) {
    return true;
  }

  auto parse_result = parser->parse_streaming_increment(delta);

  if (!parse_result.normal_text.empty()) {
    if (!send_normal_text_chunk(call_data,
                                index,
                                parse_result.normal_text,
                                request_id,
                                created_time,
                                model)) {
      return false;
    }
  }

  for (const auto& call_item : parse_result.calls) {
    stream_parser->set_has_tool_call(index, true);

    std::string tool_call_id;
    std::string function_name;
    if (call_item.name.has_value()) {
      tool_call_id = xllm::function_call::utils::generate_tool_call_id();
      function_name = call_item.name.value();
    }

    if (!send_tool_call_chunk(call_data,
                              index,
                              tool_call_id,
                              function_name,
                              call_item.parameters,
                              call_item.tool_index,
                              request_id,
                              created_time,
                              model)) {
      return false;
    }
  }

  return true;
}

}  // namespace

std::shared_ptr<ChatStreamParseState>
ResponseHandler::create_chat_stream_parse_state(
    const std::vector<JsonTool>& tools,
    const std::string& model,
    const std::string& tool_call_parser,
    const std::string& reasoning_parser) {
  auto state = std::make_shared<ChatStreamParseState>();
  state->stream_parser =
      create_stream_output_parser_with_xllm(tools,
                                            model,
                                            tool_call_parser,
                                            reasoning_parser,
                                            /*force_reasoning=*/false);
  return state;
}

bool ResponseHandler::send_delta_to_client(
    std::shared_ptr<ChatCallData> call_data,
    bool include_usage,
    int64_t created_time,
    const std::string& model,
    const llm::RequestOutput& output,
    std::shared_ptr<ChatStreamParseState> stream_state) {
  auto& response = call_data->response();
  auto& request_id = output.request_id;
  auto stream_parser = stream_state ? stream_state->stream_parser : nullptr;
  auto* first_message_sent =
      stream_state ? &stream_state->first_message_sent : nullptr;
  if (stream_parser && !output.outputs.empty()) {
    stream_parser->check_resize_for_index(output.outputs.size() - 1);
  }

  for (const auto& seq_output : output.outputs) {
    const auto& index = seq_output.index;
    std::string cur_text = seq_output.text;

    if (first_message_sent &&
        first_message_sent->find(index) == first_message_sent->end()) {
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
      if (!call_data->write(response)) {
        return false;
      }
      first_message_sent->insert(index);
    }

    if (!cur_text.empty() && stream_parser && stream_parser->is_reasoning()) {
      auto parser = stream_parser->get_reasoning_parser(index);
      auto result = parser->parse_stream_chunk(cur_text);
      if (result.normal_text.has_value()) {
        cur_text = result.normal_text.value();
      } else {
        cur_text = "";
      }
      if (result.reasoning_text.has_value()) {
        if (!send_reasoning_text_chunk(call_data,
                                       index,
                                       result.reasoning_text.value(),
                                       request_id,
                                       created_time,
                                       model)) {
          return false;
        }
      }
    }

    if (!cur_text.empty()) {
      if (stream_parser && stream_parser->is_tool_call()) {
        if (!process_tool_call_stream(call_data,
                                      stream_parser,
                                      index,
                                      cur_text,
                                      request_id,
                                      created_time,
                                      model)) {
          return false;
        }
      } else {
        response.Clear();
        response.set_object("chat.completion.chunk");
        response.set_id(request_id);
        response.set_created(created_time);
        response.set_model(model);
        auto* choice = response.add_choices();
        choice->set_index(index);
        set_logprobs(choice, seq_output.logprobs);
        auto* message = choice->mutable_delta();
        message->set_content(cur_text);
        if (!call_data->write(response)) {
          return false;
        }
      }
    }

    if (seq_output.finish_reason.has_value()) {
      if (stream_parser && stream_parser->get_has_tool_call(index)) {
        auto send_func = [&](const std::string& arguments, int tool_index) {
          return send_tool_call_chunk(call_data,
                                      index,
                                      "",
                                      "",
                                      arguments,
                                      tool_index,
                                      request_id,
                                      created_time,
                                      model);
        };
        if (!xllm::api_service::check_for_unstreamed_tool_args(
                stream_parser, index, send_func)) {
          return false;
        }
      }

      response.Clear();
      response.set_object("chat.completion.chunk");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(index);
      choice->mutable_delta();
      if (stream_parser && stream_parser->get_has_tool_call(index) &&
          seq_output.finish_reason.value() == "stop") {
        choice->set_finish_reason("tool_calls");
      } else {
        choice->set_finish_reason(seq_output.finish_reason.value());
      }
      if (!call_data->write(response)) {
        return false;
      }
    }
  }

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
    int64_t created_time,
    const std::string& model,
    const llm::RequestOutput& output) {
  auto& response = call_data->response();
  auto& request_id = output.request_id;

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
    int64_t created_time,
    const std::string& model,
    const llm::RequestOutput& req_output,
    const std::vector<JsonTool>& tools,
    const std::string& tool_call_parser,
    const std::string& reasoning_parser) {
  auto& response = call_data->response();
  auto& request_id = req_output.request_id;
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

    if (!output.text.empty()) {
      auto result =
          parse_chat_output_with_xllm(output.text,
                                      tools,
                                      model,
                                      output.finish_reason.value_or(""),
                                      tool_call_parser,
                                      reasoning_parser,
                                      response.GetArena());
      message->set_content(result.text);
      if (result.reasoning_content.has_value()) {
        message->set_reasoning_content(result.reasoning_content.value());
      }
      if (result.tool_calls.has_value()) {
        auto& source_tool_calls = *result.tool_calls;
        message->mutable_tool_calls()->Swap(&source_tool_calls);
      }
      if (!result.finish_reason.empty()) {
        choice->set_finish_reason(result.finish_reason);
      } else if (output.finish_reason.has_value()) {
        choice->set_finish_reason(output.finish_reason.value());
      }
    } else {
      message->set_content(output.text);
      if (output.finish_reason.has_value()) {
        choice->set_finish_reason(output.finish_reason.value());
      }
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
    int64_t created_time,
    const std::string& model,
    const llm::RequestOutput& req_output) {
  auto& response = call_data->response();
  auto& request_id = req_output.request_id;
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
