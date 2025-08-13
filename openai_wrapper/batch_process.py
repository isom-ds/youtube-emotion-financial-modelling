import os
import json
from openai import OpenAI
from tqdm.notebook import tqdm

from openai_wrapper.utils import create_batch_payload
from utils.gcs import list_from_gcs, load_from_gcs
from utils.formatters import split_list_into_chunks, remove_duplicates_dicts, load_json


class BatchProcessOpenAI:
    def __init__(
        self,
        key: str,
        query: str,
        filename: str,
        bucket_name: str = "youtube-us-tariffs2",
        model: str = "gpt-4.1-mini-2025-04-14",
    ):
        self.client = OpenAI(api_key=key)
        self.query = query
        self.filename = filename.replace("csv", "json")
        self.bucket_name = bucket_name
        self.model = model
        self.extracted = "extracted/" if model.startswith("text-embedding") else ""

        self.data = None
        self.paths = None
        self.file_ids = None
        self.batch_ids = None
        self.output_file_ids = None
        self.output = None

    def download_from_gcs(
        self,
        id_field: str,
        text_field: str,
        filter_ids: list = None,
        filter_col: str = None,
    ):
        # Get list of files
        blobnames = list_from_gcs(
            self.bucket_name,
            "-".join(self.query.split()).lower(),
            self.filename.replace("json", "csv"),
        )

        assert len(blobnames) > 0, f"No files found!"

        # Download from GCS
        all_data = []

        for blob in tqdm(
            blobnames, desc="Downloading blobs", unit="blob", total=len(blobnames)
        ):
            data = load_from_gcs(self.bucket_name, blob_name=blob)
            if data:
                # Filter if filter_ids
                if filter_ids:
                    filter_data = [
                        {"id": x[id_field], "text": x[text_field]}
                        for x in data
                        if x[filter_col] in filter_ids
                    ]
                else:
                    filter_data = [
                        {"id": x[id_field], "text": x[text_field]} for x in data
                    ]
                all_data += filter_data

        # Remove Duplicates
        self.data = remove_duplicates_dicts(all_data)

    def create_payload(self, chunk_size: int = 50_000):
        if self.model.startswith("text-embedding-"):
            # Load data from JSON files
            data = load_json(f"data/extracted/{self.filename}")
            self.data = [
                {"id": k, "text": v}
                for k, v in data.items()
                if isinstance(v, str) and len(v) > 0
            ]

        # Split data
        data_splits = split_list_into_chunks(self.data, chunk_size=chunk_size)
        paths = []

        if len(data_splits) == 1:
            path = f"data/{'-'.join(self.query.split()).lower()}_{self.filename.replace('.json', '')}.jsonl"
            create_batch_payload(
                data=data_splits[0],
                filename=path,
                search_term=f"'{self.query}'",
                model=self.model,
            )

            size_bytes = os.path.getsize(path)
            size_kb = size_bytes / 1024
            size_mb = size_kb / 1024

            for i in tqdm(
                [1], desc="Creating payloads", unit="payload", total=len(data_splits)
            ):
                assert (
                    size_mb < 200
                ), f"File size of {size_mb:.2f} MB exceeds maximum of 200 MB. Reduce the chunk size."

            paths.append(path)

        else:
            for i, lst in enumerate(
                tqdm(
                    data_splits,
                    desc="Creating payloads",
                    unit="payload",
                    total=len(data_splits),
                )
            ):
                path = f"data/{'-'.join(self.query.split()).lower()}_{self.filename.replace('.json', '')}_{i}.jsonl"
                create_batch_payload(
                    data=lst,
                    filename=path,
                    search_term=f"'{self.query}'",
                    model=self.model,
                )

                size_bytes = os.path.getsize(path)
                size_kb = size_bytes / 1024
                size_mb = size_kb / 1024

                assert (
                    size_mb < 200
                ), f"File size of {size_mb:.2f} MB exceeds maximum of 200 MB. Reduce the chunk size."

                paths.append(path)

        self.paths = paths

    def create_batch(self, create_file: bool = True, create_batch: bool = True):

        if create_file:
            file_ids = []

            for path in tqdm(
                self.paths, desc="Creating files", unit="file", total=len(self.paths)
            ):
                batch_input_file = self.client.files.create(
                    file=open(path, "rb"), purpose="batch"
                )
                file_ids.append(batch_input_file.id)

            self.file_ids = file_ids

        if create_batch and len(self.file_ids) > 0:
            batch_ids = []

            for path, file_id in tqdm(
                zip(self.paths, self.file_ids),
                desc="Creating batches",
                unit="batch",
                total=len(self.file_ids),
            ):
                if self.model.startswith("gpt-"):
                    endpoint = "/v1/chat/completions"
                elif self.model.startswith("text-embedding"):
                    endpoint = "/v1/embeddings"
                else:
                    raise ValueError(f"Unknown model type: {self.model}")

                batch_details = self.client.batches.create(
                    input_file_id=file_id,
                    endpoint=endpoint,
                    completion_window="24h",
                    metadata={"description": path},
                )
                batch_ids.append(batch_details.id)

            self.batch_ids = batch_ids

        if create_file and not create_batch:
            metadafile = f"data/{self.extracted}files_{'-'.join(self.query.split()).lower()}_{self.filename}"

            with open(metadafile, "w") as json_file:
                json.dump({"paths": self.paths, "file_ids": self.file_ids}, json_file)

            print(f"IDs saved as {metadafile}")

        else:
            metadafile = f"data/{self.extracted}metadata_{'-'.join(self.query.split()).lower()}_{self.filename}"

            with open(metadafile, "w") as json_file:
                json.dump(
                    {
                        "paths": self.paths,
                        "file_ids": self.file_ids,
                        "batch_ids": self.batch_ids,
                    },
                    json_file,
                )

            print(f"IDs saved as {metadafile}")

    def retrieve_batch(self, batch_ids: list = None, print_status: bool = True):

        if batch_ids:
            id_list = batch_ids
        else:
            try:
                id_list = self.batch_ids
            except:
                print("Provide list of Batch IDs")

        output_file_ids = []

        for batch_id in id_list:
            batch = self.client.batches.retrieve(batch_id)

            if print_status:
                print(f"{batch}\n")

            output_file_ids.append(batch.output_file_id)

        if output_file_ids and not all(item is None for item in output_file_ids):
            self.output_file_ids = output_file_ids

    def cancel_batch(self, batch_ids: list = None):

        if batch_ids:
            id_list = batch_ids
        else:
            id_list = self.batch_ids

        for batch_id in tqdm(
            id_list, desc="Cancelling Batches", unit="batch", total=len(id_list)
        ):
            self.client.batches.cancel(batch_id)

    def download_output(self, metadata_path: str = None):

        if metadata_path is None:
            metadata_path = f"data/{self.extracted}metadata_{'-'.join(self.query.split()).lower()}_{self.filename}"

        # Check batch ids exist
        if not self.output_file_ids:
            # Load existing file ids
            with open(metadata_path, "r") as file:
                data = json.load(file)

                self.file_ids = data["file_ids"]
                self.batch_ids = data["batch_ids"]

            # Get output file IDs
            self.retrieve_batch(print_status=False)

        for i, output_file_id in enumerate(
            tqdm(
                self.output_file_ids,
                desc="Downloading Outputs",
                unit="file",
                total=len(self.batch_ids),
            )
        ):
            # Download processed data
            file_response = self.client.files.content(output_file_id)

            output_file_name = f"data/{'-'.join(self.query.split()).lower()}_{self.filename.replace('.json', '')}_{i}_processed.jsonl"
            with open(output_file_name, "w", encoding="utf-8") as file:
                file.write(file_response.text)

            print(f"Output saved as {output_file_name}")

    def process_batch(
        self,
        id_field: str = None,
        text_field: str = None,
        filter_ids: list = None,
        filter_col: str = None,
        chunk_size: int = 50_000,
        payload: bool = True,
        batch: bool = True,
    ):

        if self.model.startswith("gpt-"):
            assert (
                id_field is not None and text_field is not None
            ), "id_field and text_field are required for gpt models."
            self.download_from_gcs(
                id_field=id_field,
                text_field=text_field,
                filter_ids=filter_ids,
                filter_col=filter_col,
            )
        # Create payloads
        if payload:
            self.create_payload(
                chunk_size=chunk_size,
            )
        # Upload files and create batches
        if batch:
            self.create_batch()

    def cleanup_openai(
        self,
        input_files: bool = False,
        batches: bool = False,
        output_files: bool = False,
    ):

        assert all(
            isinstance(x, bool) for x in [input_files, batches, output_files]
        ), "Boolean inputs only (True or False)"
        assert any([input_files, batches, output_files]), "No item selected for cleanup"

        if input_files:
            for file_id in tqdm(
                self.file_ids,
                desc="Deleting Input Files",
                unit="file",
                total=len(self.file_ids),
            ):
                self.client.files.delete(file_id)
        if batches:
            for batch_id in tqdm(
                self.batch_ids,
                desc="Deleting Batches",
                unit="batch",
                total=len(self.batch_ids),
            ):
                self.client.batches.cancel(batch_id)
        if output_files:
            for file_id in tqdm(
                self.output_file_ids,
                desc="Deleting Output Files",
                unit="file",
                total=len(self.output_file_ids),
            ):
                self.client.files.delete(file_id)
