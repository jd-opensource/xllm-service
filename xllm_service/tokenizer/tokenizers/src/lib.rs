// Import the needed libraries
use std::ffi::{c_char, CStr};
use tokenizers::tokenizer::Tokenizer;

// ported from https://github.com/mlc-ai/tokenizers-cpp

pub struct TokenizerWrapper {
    // The tokenizer
    tokenizer: Tokenizer,
    // Holds the encoded ids to avoid dropping them
    encode_ids: Vec<u32>,
    // Holds the decoded string to avoid dropping it
    decode_str: String,
    // Holds the result of the token_to_id function
    id_to_token_result: String,
}

impl TokenizerWrapper {
    pub fn encode(&mut self, text: &str, add_special_tokens: bool) {
        // Encode the text and store the ids
        self.encode_ids = Vec::from(
            self.tokenizer
                .encode(text, add_special_tokens)
                .unwrap()
                .get_ids(),
        );
    }

    pub fn decode(&mut self, ids: Vec<u32>, skip_special_tokens: bool) {
        // Decode the ids and store the string
        self.decode_str = self.tokenizer.decode(&ids, skip_special_tokens).unwrap();
    }

    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.tokenizer.get_vocab_size(with_added_tokens)
    }
}

#[no_mangle]
extern "C" fn tokenizer_from_file(path: *const c_char) -> *mut TokenizerWrapper {
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => panic!("Failed to convert C string to Rust string"),
    };

    let boxed = Box::new(TokenizerWrapper {
        tokenizer: Tokenizer::from_file(path_str).unwrap().into(),
        encode_ids: Vec::new(),
        decode_str: String::new(),
        id_to_token_result: String::new(),
    });

    Box::into_raw(boxed)
}

#[no_mangle]
extern "C" fn tokenizer_encode(
    handle: *mut TokenizerWrapper,
    input_cstr: *const u8,
    len: usize,
    add_special_tokens: bool,
) {
    unsafe {
        let input_data = std::str::from_utf8(std::slice::from_raw_parts(input_cstr, len)).unwrap();
        (*handle).encode(input_data, add_special_tokens);
    }
}

#[no_mangle]
extern "C" fn tokenizer_get_encode_ids(
    handle: *mut TokenizerWrapper,
    out_data: *mut *mut u32,
    out_len: *mut usize,
) {
    unsafe {
        *out_data = (*handle).encode_ids.as_mut_ptr();
        *out_len = (*handle).encode_ids.len()
    }
}

#[no_mangle]
extern "C" fn tokenizer_decode(
    handle: *mut TokenizerWrapper,
    input_ids: *const u32,
    len: usize,
    skip_special_tokens: bool,
) {
    unsafe {
        let input_data = Vec::from(std::slice::from_raw_parts(input_ids, len));
        (*handle).decode(input_data, skip_special_tokens);
    }
}

#[no_mangle]
extern "C" fn tokenizer_get_decode_str(
    handle: *mut TokenizerWrapper,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        *out_cstr = (*handle).decode_str.as_mut_ptr();
        *out_len = (&(*handle).decode_str).len();
    }
}

#[no_mangle]
extern "C" fn tokenizer_free(wrapper: *mut TokenizerWrapper) {
    unsafe {
        drop(Box::from_raw(wrapper));
    }
}

#[no_mangle]
extern "C" fn tokenizer_token_to_id(
    handle: *mut TokenizerWrapper,
    token: *const u8,
    len: usize
) {
    unsafe {
        let token: &str = std::str::from_utf8(std::slice::from_raw_parts(token, len)).unwrap();
        let id = (*handle).tokenizer.token_to_id(token);
        match id {
            Some(id) => id as i32,
            None => -1,
        };
    }
}

#[no_mangle]
extern "C" fn tokenizer_id_to_token(
    handle: *mut TokenizerWrapper,
    id: u32,
    out_cstr: *mut *mut u8,
    out_len: *mut usize,
) {
    unsafe {
        let str = (*handle).tokenizer.id_to_token(id);
        (*handle).id_to_token_result = match str {
            Some(s) => s,
            None => String::from(""),
        };

        *out_cstr = (*handle).id_to_token_result.as_mut_ptr();
        *out_len = (&(*handle).id_to_token_result).len();
    }
}

#[no_mangle]
extern "C" fn tokenizer_get_vocab_size(
    handle: *mut TokenizerWrapper, 
    with_added_tokens: bool) -> usize {
    unsafe {
        (*handle).get_vocab_size(with_added_tokens)
    }
}
