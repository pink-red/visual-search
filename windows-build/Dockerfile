FROM tobix/pywine:3.11

RUN apt update && apt install p7zip wget

WORKDIR /workdir

COPY ./entrypoint.sh entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
