# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: data.proto

import sys

_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor.FileDescriptor(
    name='data.proto',
    package='autodl',
    syntax='proto2',
    serialized_pb=_b(
        '\n\ndata.proto\x12\x06\x61utodl\"\x1f\n\nDenseValue\x12\x11\n\x05value\x18\x01 \x03(\x02\x42\x02\x10\x01\"6\n\x0bSparseEntry\x12\x0b\n\x03row\x18\x01 \x01(\x05\x12\x0b\n\x03\x63ol\x18\x02 \x01(\x05\x12\r\n\x05value\x18\x03 \x01(\x02\"#\n\nCompressed\x12\x15\n\rencoded_image\x18\x01 \x01(\x0c\"1\n\x0bSparseValue\x12\"\n\x05\x65ntry\x18\x01 \x03(\x0b\x32\x13.autodl.SparseEntry\"\xac\x02\n\nMatrixSpec\x12\x11\n\tcol_count\x18\x01 \x01(\x05\x12\x11\n\trow_count\x18\x02 \x01(\x05\x12\x17\n\x0fis_sequence_col\x18\x03 \x01(\x08\x12\x17\n\x0fis_sequence_row\x18\x04 \x01(\x08\x12\x18\n\x10has_locality_col\x18\x05 \x01(\x08\x12\x18\n\x10has_locality_row\x18\x06 \x01(\x08\x12\x30\n\x06\x66ormat\x18\x08 \x01(\x0e\x32\x19.autodl.MatrixSpec.Format:\x05\x44\x45NSE\x12\x15\n\tis_sparse\x18\x07 \x01(\x08\x42\x02\x18\x01\x12\x18\n\x0cnum_channels\x18\t \x01(\x05:\x02-1\"/\n\x06\x46ormat\x12\t\n\x05\x44\x45NSE\x10\x00\x12\n\n\x06SPARSE\x10\x01\x12\x0e\n\nCOMPRESSED\x10\x02\"\xc0\x01\n\x06Matrix\x12%\n\x06sparse\x18\x01 \x01(\x0b\x32\x13.autodl.SparseValueH\x00\x12#\n\x05\x64\x65nse\x18\x02 \x01(\x0b\x32\x12.autodl.DenseValueH\x00\x12(\n\ncompressed\x18\x05 \x01(\x0b\x32\x12.autodl.CompressedH\x00\x12 \n\x04spec\x18\x03 \x01(\x0b\x32\x12.autodl.MatrixSpec\x12\x14\n\x0c\x62undle_index\x18\x04 \x01(\x05\x42\x08\n\x06values\"F\n\x0cMatrixBundle\x12\x1e\n\x06matrix\x18\x01 \x03(\x0b\x32\x0e.autodl.Matrix\x12\x16\n\x0esequence_index\x18\x02 \x01(\x05\"B\n\x05Input\x12$\n\x06\x62undle\x18\x01 \x03(\x0b\x32\x14.autodl.MatrixBundle\x12\x13\n\x0bis_sequence\x18\x02 \x01(\x08\"%\n\x05Label\x12\r\n\x05index\x18\x01 \x01(\x05\x12\r\n\x05score\x18\x02 \x01(\x02\"&\n\x06Output\x12\x1c\n\x05label\x18\x01 \x03(\x0b\x32\r.autodl.Label\"R\n\x06Sample\x12\n\n\x02id\x18\x01 \x01(\x05\x12\x1c\n\x05input\x18\x02 \x01(\x0b\x32\r.autodl.Input\x12\x1e\n\x06output\x18\x03 \x01(\x0b\x32\x0e.autodl.Output\"\xad\x04\n\x11\x44\x61taSpecification\x12\'\n\x0bmatrix_spec\x18\x01 \x03(\x0b\x32\x12.autodl.MatrixSpec\x12\x13\n\x0bis_sequence\x18\x02 \x01(\x08\x12\x12\n\noutput_dim\x18\x03 \x01(\x05\x12J\n\x12label_to_index_map\x18\x04 \x03(\x0b\x32..autodl.DataSpecification.LabelToIndexMapEntry\x12N\n\x14\x66\x65\x61ture_to_index_map\x18\x05 \x03(\x0b\x32\x30.autodl.DataSpecification.FeatureToIndexMapEntry\x12N\n\x14\x63hannel_to_index_map\x18\x06 \x03(\x0b\x32\x30.autodl.DataSpecification.ChannelToIndexMapEntry\x12\x18\n\rsequence_size\x18\x07 \x01(\x05:\x01\x31\x12\x14\n\x0csample_count\x18\x08 \x01(\x05\x1a\x36\n\x14LabelToIndexMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x38\n\x16\x46\x65\x61tureToIndexMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x38\n\x16\x43hannelToIndexMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01')
)

_MATRIXSPEC_FORMAT = _descriptor.EnumDescriptor(
    name='Format',
    full_name='autodl.MatrixSpec.Format',
    filename=None,
    file=DESCRIPTOR,
    values=[
        _descriptor.EnumValueDescriptor(
            name='DENSE', index=0, number=0,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='SPARSE', index=1, number=1,
            options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='COMPRESSED', index=2, number=2,
            options=None,
            type=None),
    ],
    containing_type=None,
    options=None,
    serialized_start=453,
    serialized_end=500,
)
_sym_db.RegisterEnumDescriptor(_MATRIXSPEC_FORMAT)

_DENSEVALUE = _descriptor.Descriptor(
    name='DenseValue',
    full_name='autodl.DenseValue',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='value', full_name='autodl.DenseValue.value', index=0,
            number=1, type=2, cpp_type=6, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001')), file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=22,
    serialized_end=53,
)

_SPARSEENTRY = _descriptor.Descriptor(
    name='SparseEntry',
    full_name='autodl.SparseEntry',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='row', full_name='autodl.SparseEntry.row', index=0,
            number=1, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='col', full_name='autodl.SparseEntry.col', index=1,
            number=2, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='value', full_name='autodl.SparseEntry.value', index=2,
            number=3, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=55,
    serialized_end=109,
)

_COMPRESSED = _descriptor.Descriptor(
    name='Compressed',
    full_name='autodl.Compressed',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='encoded_image', full_name='autodl.Compressed.encoded_image', index=0,
            number=1, type=12, cpp_type=9, label=1,
            has_default_value=False, default_value=_b(""),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=111,
    serialized_end=146,
)

_SPARSEVALUE = _descriptor.Descriptor(
    name='SparseValue',
    full_name='autodl.SparseValue',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='entry', full_name='autodl.SparseValue.entry', index=0,
            number=1, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=148,
    serialized_end=197,
)

_MATRIXSPEC = _descriptor.Descriptor(
    name='MatrixSpec',
    full_name='autodl.MatrixSpec',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='col_count', full_name='autodl.MatrixSpec.col_count', index=0,
            number=1, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='row_count', full_name='autodl.MatrixSpec.row_count', index=1,
            number=2, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='is_sequence_col', full_name='autodl.MatrixSpec.is_sequence_col', index=2,
            number=3, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='is_sequence_row', full_name='autodl.MatrixSpec.is_sequence_row', index=3,
            number=4, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='has_locality_col', full_name='autodl.MatrixSpec.has_locality_col', index=4,
            number=5, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='has_locality_row', full_name='autodl.MatrixSpec.has_locality_row', index=5,
            number=6, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='format', full_name='autodl.MatrixSpec.format', index=6,
            number=8, type=14, cpp_type=8, label=1,
            has_default_value=True, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='is_sparse', full_name='autodl.MatrixSpec.is_sparse', index=7,
            number=7, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\030\001')), file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='num_channels', full_name='autodl.MatrixSpec.num_channels', index=8,
            number=9, type=5, cpp_type=1, label=1,
            has_default_value=True, default_value=-1,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
        _MATRIXSPEC_FORMAT,
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=200,
    serialized_end=500,
)

_MATRIX = _descriptor.Descriptor(
    name='Matrix',
    full_name='autodl.Matrix',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='sparse', full_name='autodl.Matrix.sparse', index=0,
            number=1, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='dense', full_name='autodl.Matrix.dense', index=1,
            number=2, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='compressed', full_name='autodl.Matrix.compressed', index=2,
            number=5, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='spec', full_name='autodl.Matrix.spec', index=3,
            number=3, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='bundle_index', full_name='autodl.Matrix.bundle_index', index=4,
            number=4, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
        _descriptor.OneofDescriptor(
            name='values', full_name='autodl.Matrix.values',
            index=0, containing_type=None, fields=[]),
    ],
    serialized_start=503,
    serialized_end=695,
)

_MATRIXBUNDLE = _descriptor.Descriptor(
    name='MatrixBundle',
    full_name='autodl.MatrixBundle',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='matrix', full_name='autodl.MatrixBundle.matrix', index=0,
            number=1, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='sequence_index', full_name='autodl.MatrixBundle.sequence_index', index=1,
            number=2, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=697,
    serialized_end=767,
)

_INPUT = _descriptor.Descriptor(
    name='Input',
    full_name='autodl.Input',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='bundle', full_name='autodl.Input.bundle', index=0,
            number=1, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='is_sequence', full_name='autodl.Input.is_sequence', index=1,
            number=2, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=769,
    serialized_end=835,
)

_LABEL = _descriptor.Descriptor(
    name='Label',
    full_name='autodl.Label',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='index', full_name='autodl.Label.index', index=0,
            number=1, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='score', full_name='autodl.Label.score', index=1,
            number=2, type=2, cpp_type=6, label=1,
            has_default_value=False, default_value=float(0),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=837,
    serialized_end=874,
)

_OUTPUT = _descriptor.Descriptor(
    name='Output',
    full_name='autodl.Output',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='label', full_name='autodl.Output.label', index=0,
            number=1, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=876,
    serialized_end=914,
)

_SAMPLE = _descriptor.Descriptor(
    name='Sample',
    full_name='autodl.Sample',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='id', full_name='autodl.Sample.id', index=0,
            number=1, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='input', full_name='autodl.Sample.input', index=1,
            number=2, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='output', full_name='autodl.Sample.output', index=2,
            number=3, type=11, cpp_type=10, label=1,
            has_default_value=False, default_value=None,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=916,
    serialized_end=998,
)

_DATASPECIFICATION_LABELTOINDEXMAPENTRY = _descriptor.Descriptor(
    name='LabelToIndexMapEntry',
    full_name='autodl.DataSpecification.LabelToIndexMapEntry',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='key', full_name='autodl.DataSpecification.LabelToIndexMapEntry.key', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=_b("").decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='value', full_name='autodl.DataSpecification.LabelToIndexMapEntry.value', index=1,
            number=2, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1388,
    serialized_end=1442,
)

_DATASPECIFICATION_FEATURETOINDEXMAPENTRY = _descriptor.Descriptor(
    name='FeatureToIndexMapEntry',
    full_name='autodl.DataSpecification.FeatureToIndexMapEntry',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='key', full_name='autodl.DataSpecification.FeatureToIndexMapEntry.key', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=_b("").decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='value', full_name='autodl.DataSpecification.FeatureToIndexMapEntry.value', index=1,
            number=2, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1444,
    serialized_end=1500,
)

_DATASPECIFICATION_CHANNELTOINDEXMAPENTRY = _descriptor.Descriptor(
    name='ChannelToIndexMapEntry',
    full_name='autodl.DataSpecification.ChannelToIndexMapEntry',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='key', full_name='autodl.DataSpecification.ChannelToIndexMapEntry.key', index=0,
            number=1, type=9, cpp_type=9, label=1,
            has_default_value=False, default_value=_b("").decode('utf-8'),
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='value', full_name='autodl.DataSpecification.ChannelToIndexMapEntry.value', index=1,
            number=2, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1502,
    serialized_end=1558,
)

_DATASPECIFICATION = _descriptor.Descriptor(
    name='DataSpecification',
    full_name='autodl.DataSpecification',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='matrix_spec', full_name='autodl.DataSpecification.matrix_spec', index=0,
            number=1, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='is_sequence', full_name='autodl.DataSpecification.is_sequence', index=1,
            number=2, type=8, cpp_type=7, label=1,
            has_default_value=False, default_value=False,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='output_dim', full_name='autodl.DataSpecification.output_dim', index=2,
            number=3, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='label_to_index_map', full_name='autodl.DataSpecification.label_to_index_map', index=3,
            number=4, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='feature_to_index_map', full_name='autodl.DataSpecification.feature_to_index_map', index=4,
            number=5, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='channel_to_index_map', full_name='autodl.DataSpecification.channel_to_index_map', index=5,
            number=6, type=11, cpp_type=10, label=3,
            has_default_value=False, default_value=[],
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='sequence_size', full_name='autodl.DataSpecification.sequence_size', index=6,
            number=7, type=5, cpp_type=1, label=1,
            has_default_value=True, default_value=1,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
        _descriptor.FieldDescriptor(
            name='sample_count', full_name='autodl.DataSpecification.sample_count', index=7,
            number=8, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None, file=DESCRIPTOR),
    ],
    extensions=[
    ],
    nested_types=[_DATASPECIFICATION_LABELTOINDEXMAPENTRY, _DATASPECIFICATION_FEATURETOINDEXMAPENTRY,
                  _DATASPECIFICATION_CHANNELTOINDEXMAPENTRY, ],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto2',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=1001,
    serialized_end=1558,
)

_SPARSEVALUE.fields_by_name['entry'].message_type = _SPARSEENTRY
_MATRIXSPEC.fields_by_name['format'].enum_type = _MATRIXSPEC_FORMAT
_MATRIXSPEC_FORMAT.containing_type = _MATRIXSPEC
_MATRIX.fields_by_name['sparse'].message_type = _SPARSEVALUE
_MATRIX.fields_by_name['dense'].message_type = _DENSEVALUE
_MATRIX.fields_by_name['compressed'].message_type = _COMPRESSED
_MATRIX.fields_by_name['spec'].message_type = _MATRIXSPEC
_MATRIX.oneofs_by_name['values'].fields.append(
    _MATRIX.fields_by_name['sparse'])
_MATRIX.fields_by_name['sparse'].containing_oneof = _MATRIX.oneofs_by_name['values']
_MATRIX.oneofs_by_name['values'].fields.append(
    _MATRIX.fields_by_name['dense'])
_MATRIX.fields_by_name['dense'].containing_oneof = _MATRIX.oneofs_by_name['values']
_MATRIX.oneofs_by_name['values'].fields.append(
    _MATRIX.fields_by_name['compressed'])
_MATRIX.fields_by_name['compressed'].containing_oneof = _MATRIX.oneofs_by_name['values']
_MATRIXBUNDLE.fields_by_name['matrix'].message_type = _MATRIX
_INPUT.fields_by_name['bundle'].message_type = _MATRIXBUNDLE
_OUTPUT.fields_by_name['label'].message_type = _LABEL
_SAMPLE.fields_by_name['input'].message_type = _INPUT
_SAMPLE.fields_by_name['output'].message_type = _OUTPUT
_DATASPECIFICATION_LABELTOINDEXMAPENTRY.containing_type = _DATASPECIFICATION
_DATASPECIFICATION_FEATURETOINDEXMAPENTRY.containing_type = _DATASPECIFICATION
_DATASPECIFICATION_CHANNELTOINDEXMAPENTRY.containing_type = _DATASPECIFICATION
_DATASPECIFICATION.fields_by_name['matrix_spec'].message_type = _MATRIXSPEC
_DATASPECIFICATION.fields_by_name['label_to_index_map'].message_type = _DATASPECIFICATION_LABELTOINDEXMAPENTRY
_DATASPECIFICATION.fields_by_name['feature_to_index_map'].message_type = _DATASPECIFICATION_FEATURETOINDEXMAPENTRY
_DATASPECIFICATION.fields_by_name['channel_to_index_map'].message_type = _DATASPECIFICATION_CHANNELTOINDEXMAPENTRY
DESCRIPTOR.message_types_by_name['DenseValue'] = _DENSEVALUE
DESCRIPTOR.message_types_by_name['SparseEntry'] = _SPARSEENTRY
DESCRIPTOR.message_types_by_name['Compressed'] = _COMPRESSED
DESCRIPTOR.message_types_by_name['SparseValue'] = _SPARSEVALUE
DESCRIPTOR.message_types_by_name['MatrixSpec'] = _MATRIXSPEC
DESCRIPTOR.message_types_by_name['Matrix'] = _MATRIX
DESCRIPTOR.message_types_by_name['MatrixBundle'] = _MATRIXBUNDLE
DESCRIPTOR.message_types_by_name['Input'] = _INPUT
DESCRIPTOR.message_types_by_name['Label'] = _LABEL
DESCRIPTOR.message_types_by_name['Output'] = _OUTPUT
DESCRIPTOR.message_types_by_name['Sample'] = _SAMPLE
DESCRIPTOR.message_types_by_name['DataSpecification'] = _DATASPECIFICATION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DenseValue = _reflection.GeneratedProtocolMessageType('DenseValue', (_message.Message,), dict(
    DESCRIPTOR=_DENSEVALUE,
    __module__='data_pb2'
    # @@protoc_insertion_point(class_scope:autodl.DenseValue)
))
_sym_db.RegisterMessage(DenseValue)

SparseEntry = _reflection.GeneratedProtocolMessageType('SparseEntry', (_message.Message,), dict(
    DESCRIPTOR=_SPARSEENTRY,
    __module__='data_pb2'
    # @@protoc_insertion_point(class_scope:autodl.SparseEntry)
))
_sym_db.RegisterMessage(SparseEntry)

Compressed = _reflection.GeneratedProtocolMessageType('Compressed', (_message.Message,), dict(
    DESCRIPTOR=_COMPRESSED,
    __module__='data_pb2'
    # @@protoc_insertion_point(class_scope:autodl.Compressed)
))
_sym_db.RegisterMessage(Compressed)

SparseValue = _reflection.GeneratedProtocolMessageType('SparseValue', (_message.Message,), dict(
    DESCRIPTOR=_SPARSEVALUE,
    __module__='data_pb2'
    # @@protoc_insertion_point(class_scope:autodl.SparseValue)
))
_sym_db.RegisterMessage(SparseValue)

MatrixSpec = _reflection.GeneratedProtocolMessageType('MatrixSpec', (_message.Message,), dict(
    DESCRIPTOR=_MATRIXSPEC,
    __module__='data_pb2'
    # @@protoc_insertion_point(class_scope:autodl.MatrixSpec)
))
_sym_db.RegisterMessage(MatrixSpec)

Matrix = _reflection.GeneratedProtocolMessageType('Matrix', (_message.Message,), dict(
    DESCRIPTOR=_MATRIX,
    __module__='data_pb2'
    # @@protoc_insertion_point(class_scope:autodl.Matrix)
))
_sym_db.RegisterMessage(Matrix)

MatrixBundle = _reflection.GeneratedProtocolMessageType('MatrixBundle', (_message.Message,), dict(
    DESCRIPTOR=_MATRIXBUNDLE,
    __module__='data_pb2'
    # @@protoc_insertion_point(class_scope:autodl.MatrixBundle)
))
_sym_db.RegisterMessage(MatrixBundle)

Input = _reflection.GeneratedProtocolMessageType('Input', (_message.Message,), dict(
    DESCRIPTOR=_INPUT,
    __module__='data_pb2'
    # @@protoc_insertion_point(class_scope:autodl.Input)
))
_sym_db.RegisterMessage(Input)

Label = _reflection.GeneratedProtocolMessageType('Label', (_message.Message,), dict(
    DESCRIPTOR=_LABEL,
    __module__='data_pb2'
    # @@protoc_insertion_point(class_scope:autodl.Label)
))
_sym_db.RegisterMessage(Label)

Output = _reflection.GeneratedProtocolMessageType('Output', (_message.Message,), dict(
    DESCRIPTOR=_OUTPUT,
    __module__='data_pb2'
    # @@protoc_insertion_point(class_scope:autodl.Output)
))
_sym_db.RegisterMessage(Output)

Sample = _reflection.GeneratedProtocolMessageType('Sample', (_message.Message,), dict(
    DESCRIPTOR=_SAMPLE,
    __module__='data_pb2'
    # @@protoc_insertion_point(class_scope:autodl.Sample)
))
_sym_db.RegisterMessage(Sample)

DataSpecification = _reflection.GeneratedProtocolMessageType('DataSpecification', (_message.Message,), dict(

    LabelToIndexMapEntry=_reflection.GeneratedProtocolMessageType('LabelToIndexMapEntry', (_message.Message,), dict(
        DESCRIPTOR=_DATASPECIFICATION_LABELTOINDEXMAPENTRY,
        __module__='data_pb2'
        # @@protoc_insertion_point(class_scope:autodl.DataSpecification.LabelToIndexMapEntry)
    ))
    ,

    FeatureToIndexMapEntry=_reflection.GeneratedProtocolMessageType('FeatureToIndexMapEntry', (_message.Message,), dict(
        DESCRIPTOR=_DATASPECIFICATION_FEATURETOINDEXMAPENTRY,
        __module__='data_pb2'
        # @@protoc_insertion_point(class_scope:autodl.DataSpecification.FeatureToIndexMapEntry)
    ))
    ,

    ChannelToIndexMapEntry=_reflection.GeneratedProtocolMessageType('ChannelToIndexMapEntry', (_message.Message,), dict(
        DESCRIPTOR=_DATASPECIFICATION_CHANNELTOINDEXMAPENTRY,
        __module__='data_pb2'
        # @@protoc_insertion_point(class_scope:autodl.DataSpecification.ChannelToIndexMapEntry)
    ))
    ,
    DESCRIPTOR=_DATASPECIFICATION,
    __module__='data_pb2'
    # @@protoc_insertion_point(class_scope:autodl.DataSpecification)
))
_sym_db.RegisterMessage(DataSpecification)
_sym_db.RegisterMessage(DataSpecification.LabelToIndexMapEntry)
_sym_db.RegisterMessage(DataSpecification.FeatureToIndexMapEntry)
_sym_db.RegisterMessage(DataSpecification.ChannelToIndexMapEntry)

_DENSEVALUE.fields_by_name['value'].has_options = True
_DENSEVALUE.fields_by_name['value']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_MATRIXSPEC.fields_by_name['is_sparse'].has_options = True
_MATRIXSPEC.fields_by_name['is_sparse']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(),
                                                                             _b('\030\001'))
_DATASPECIFICATION_LABELTOINDEXMAPENTRY.has_options = True
_DATASPECIFICATION_LABELTOINDEXMAPENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(),
                                                                             _b('8\001'))
_DATASPECIFICATION_FEATURETOINDEXMAPENTRY.has_options = True
_DATASPECIFICATION_FEATURETOINDEXMAPENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(),
                                                                               _b('8\001'))
_DATASPECIFICATION_CHANNELTOINDEXMAPENTRY.has_options = True
_DATASPECIFICATION_CHANNELTOINDEXMAPENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(),
                                                                               _b('8\001'))
# @@protoc_insertion_point(module_scope)

