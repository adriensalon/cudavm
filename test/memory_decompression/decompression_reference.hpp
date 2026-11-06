#pragma once
namespace cudavm {
// A fixture validator only: this decoder is never linked into the facade.
struct decompression_bits {
    std::span<const unsigned char> input; size_t position{};
    unsigned read(unsigned count) {
        require(count <= 24 && position <= input.size() * 8 && count <= input.size() * 8 - position, "truncated compression fixture");
        unsigned value{}; for (unsigned i = 0; i < count; ++i, ++position) value |= unsigned((input[position / 8] >> (position % 8)) & 1) << i;
        return value;
    }
    bool end() const { return position == input.size() * 8; }
    void align() { position = (position + 7) / 8 * 8; }
};
constexpr size_t decompression_reference_limit = 1024 * 1024;
void decompression_literals(decompression_bits& bits, std::vector<unsigned char>& result, size_t count) {
    require(count <= decompression_reference_limit - result.size(), "fixture literal bound");
    for (size_t i = 0; i < count; ++i) result.push_back(static_cast<unsigned char>(bits.read(8)));
}
void decompression_match(std::vector<unsigned char>& result, size_t distance, size_t length) {
    require(distance && distance <= result.size() && length <= decompression_reference_limit - result.size(), "fixture match range");
    while (length--) result.push_back(result[result.size() - distance]);
}
struct decompression_huffman {
    struct node { int children[2]{-1, -1}; int symbol = -1; };
    std::vector<node> nodes{node{}};
    explicit decompression_huffman(std::span<const unsigned> lengths) {
        std::array<unsigned, 16> counts{}, next{};
        for (auto length : lengths) { require(length < counts.size(), "fixture Huffman length"); if (length) ++counts[length]; }
        unsigned code{};
        for (unsigned length = 1; length < counts.size(); ++length) {
            code = (code + counts[length - 1]) << 1; next[length] = code;
            require(code + counts[length] <= (1u << length), "oversubscribed fixture Huffman alphabet");
        }
        for (unsigned symbol = 0; symbol < lengths.size(); ++symbol) if (auto length = lengths[symbol]) {
            const auto value = next[length]++; unsigned current{};
            for (unsigned bit = length; bit; --bit) {
                require(nodes[current].symbol < 0, "overlapping fixture Huffman codes");
                auto direction = (value >> (bit - 1)) & 1; auto child = nodes[current].children[direction];
                if (child < 0) { child = int(nodes.size()); nodes[current].children[direction] = child; nodes.emplace_back(); }
                current = unsigned(child);
            }
            require(nodes[current].symbol < 0, "duplicate fixture Huffman code"); nodes[current].symbol = int(symbol);
        }
    }
    unsigned decode(decompression_bits& bits) const {
        unsigned current{};
        for (unsigned depth = 0; depth <= 15; ++depth) {
            if (nodes[current].symbol >= 0) return unsigned(nodes[current].symbol);
            const auto child = nodes[current].children[bits.read(1)]; require(child >= 0, "invalid fixture Huffman code"); current = unsigned(child);
        }
        throw std::runtime_error("excessive fixture Huffman depth");
    }
};
std::vector<unsigned char> decompression_reference_deflate(std::span<const unsigned char> input) {
    decompression_bits bits{input}; std::vector<unsigned char> result; bool final{};
    constexpr unsigned length_base[]{3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
    constexpr unsigned length_extra[]{0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
    constexpr unsigned distance_base[]{1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
    constexpr unsigned distance_extra[]{0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
    while (!final) {
        final = bits.read(1) != 0; auto type = bits.read(2);
        if (!type) {
            bits.align(); const auto count = bits.read(16), complement = bits.read(16);
            require((count ^ complement) == 0xffff, "fixture stored-block length complement"); decompression_literals(bits, result, count); continue;
        }
        require(type <= 2, "reserved fixture Deflate block type"); std::vector<unsigned> literal_lengths, distance_lengths;
        if (type == 1) {
            literal_lengths.resize(288); distance_lengths.assign(32, 5);
            for (unsigned i = 0; i < literal_lengths.size(); ++i) literal_lengths[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
        } else {
            const auto literal_count = bits.read(5) + 257, distance_count = bits.read(5) + 1, code_count = bits.read(4) + 4;
            constexpr unsigned order[]{16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
            std::array<unsigned, 19> code_lengths{}; for (unsigned i = 0; i < code_count; ++i) code_lengths[order[i]] = bits.read(3);
            decompression_huffman codes(code_lengths); std::vector<unsigned> lengths;
            while (lengths.size() < literal_count + distance_count) {
                auto symbol = codes.decode(bits); if (symbol < 16) { lengths.push_back(symbol); continue; }
                unsigned count{}, value{};
                if (symbol == 16) { require(!lengths.empty(), "fixture repeated length has a predecessor"); value = lengths.back(); count = bits.read(2) + 3; }
                else if (symbol == 17) count = bits.read(3) + 3;
                else { require(symbol == 18, "fixture length alphabet symbol"); count = bits.read(7) + 11; }
                require(count <= literal_count + distance_count - lengths.size(), "fixture code-length repeat bound"); lengths.insert(lengths.end(), count, value);
            }
            literal_lengths.assign(lengths.begin(), lengths.begin() + literal_count); distance_lengths.assign(lengths.begin() + literal_count, lengths.end());
        }
        require(literal_lengths[256] != 0, "fixture Deflate end symbol exists");
        decompression_huffman literals(literal_lengths), distances(distance_lengths);
        for (;;) {
            auto symbol = literals.decode(bits);
            if (symbol < 256) { require(result.size() < decompression_reference_limit, "fixture decoded bound"); result.push_back(static_cast<unsigned char>(symbol)); continue; }
            if (symbol == 256) break;
            require(symbol <= 285, "fixture Deflate length symbol"); auto length = length_base[symbol - 257] + bits.read(length_extra[symbol - 257]);
            auto distance = distances.decode(bits); require(distance < 30, "fixture Deflate distance symbol");
            decompression_match(result, distance_base[distance] + bits.read(distance_extra[distance]), length);
        }
    }
    bits.align(); require(bits.end(), "fixture Deflate consumes its exact byte extent"); return result;
}
std::vector<unsigned char> decompression_reference_snappy(std::span<const unsigned char> input) {
    decompression_bits bits{input}; std::vector<unsigned char> result; size_t declared{};
    for (unsigned shift = 0;; shift += 7) {
        require(shift <= 28, "fixture Snappy size varint"); auto byte = bits.read(8); declared |= size_t(byte & 127) << shift; if (!(byte & 128)) break;
    }
    require(declared <= decompression_reference_limit, "fixture Snappy output bound");
    while (!bits.end()) {
        const auto tag = bits.read(8), kind = tag & 3;
        if (!kind) {
            uint64_t length = tag >> 2;
            if (length >= 60) {
                auto bytes = unsigned(length - 59); length = 0;
                for (unsigned i = 0; i < bytes; ++i) length |= uint64_t(bits.read(8)) << (i * 8);
            }
            require(length < decompression_reference_limit, "fixture Snappy literal extent"); decompression_literals(bits, result, size_t(length + 1));
        } else {
            unsigned length{}, distance{};
            if (kind == 1) { length = 4 + ((tag >> 2) & 7); distance = ((tag & 0xe0) << 3) | bits.read(8); }
            else {
                length = 1 + (tag >> 2); for (unsigned i = 0; i < (kind == 2 ? 2u : 4u); ++i) distance |= bits.read(8) << (i * 8);
            }
            decompression_match(result, distance, length);
        }
    }
    require(result.size() == declared, "fixture Snappy declared size"); return result;
}
std::vector<unsigned char> decompression_reference_lz4(std::span<const unsigned char> input) {
    decompression_bits bits{input}; std::vector<unsigned char> result;
    auto length = [&](size_t value) {
        if (value == 15) { unsigned byte{}; do { byte = bits.read(8); value += byte; require(value <= decompression_reference_limit, "fixture LZ4 extended length"); } while (byte == 255); }
        return value;
    };
    while (!bits.end()) {
        const auto token = bits.read(8); decompression_literals(bits, result, length(token >> 4)); if (bits.end()) break;
        const auto distance = bits.read(16); decompression_match(result, distance, length(token & 15) + 4);
    }
    return result;
}
std::vector<unsigned char> decompression_reference(CUmemDecompressAlgorithm algorithm, std::span<const unsigned char> input) {
    if (algorithm == CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE) return decompression_reference_deflate(input);
    if (algorithm == CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY) return decompression_reference_snappy(input);
    require(algorithm == CU_MEM_DECOMPRESS_ALGORITHM_LZ4, "fixture algorithm"); return decompression_reference_lz4(input);
}
}
